import yaml
import logging
import gc
from copy import deepcopy
import functools
import copy
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
import threading
from abc import ABC, abstractmethod
import builtins
from contextlib import contextmanager

from constants import CHAT_MODELS, system_message, GLM4Z1_CHAT_TEMPLATE
from utilities import my_cprint, has_bfloat16_support

# logging.getLogger("transformers").setLevel(logging.WARNING) # adjust to see deprecation and other non-fatal errors
logging.getLogger("transformers").setLevel(logging.ERROR)

@contextmanager
def utf8_file_operations():
    """Context manager that ensures all file operations use UTF-8 encoding by default."""
    original_open = builtins.open
    
    def utf8_open(path, *args, **kwargs):
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        return original_open(path, *args, **kwargs)

    builtins.open = utf8_open
    try:
        yield
    finally:
        builtins.open = original_open

def _configure_device_settings(settings, model_info):
    """
    Configure device, dtype, and quantization based on model_info precision.
    Returns 'cuda' or 'cpu'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.setdefault('tokenizer_settings', {})
    settings.setdefault('model_settings', {})

    if device == "cuda":
        native = model_info.get("precision")
        if native in ("float32", "bfloat16"):
            dtype = torch.bfloat16 if has_bfloat16_support() else torch.float16
        else:
            dtype = torch.float16
        settings['tokenizer_settings']['torch_dtype'] = dtype
        settings['model_settings']['torch_dtype'] = dtype
        qc = settings['model_settings'].get("quantization_config")
        if qc is not None:
            qc.bnb_4bit_compute_dtype = dtype
    else:
        settings['model_settings'].pop('quantization_config', None)
        settings['model_settings']['device_map'] = "cpu"

    return device

def get_model_settings(base_settings, attn_implementation):
    settings = copy.deepcopy(base_settings)
    # settings['model_settings']['attn_implementation'] = attn_implementation
    return settings
    
def get_max_length(model_name):
    if model_name in CHAT_MODELS:
        return CHAT_MODELS[model_name].get('max_tokens', 8192)
    return 8192

def get_max_new_tokens(model_name):
    if model_name in CHAT_MODELS:
        return CHAT_MODELS[model_name].get('max_new_tokens', 1024)
    return 1024

def get_generation_settings(max_length, max_new_tokens):
    return {
        'max_length': max_length,
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
        'num_beams': 1,
        'use_cache': True,
        'temperature': None,
        'top_p': None,
        'top_k': None,
    }

bnb_bfloat16_settings = {
    'tokenizer_settings': {
        'torch_dtype': torch.bfloat16,
        # 'add_bos_token': False, # doublecheck this
    },
    'model_settings': {
        'torch_dtype': torch.bfloat16,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        'low_cpu_mem_usage': True,
        # 'attn_implementation': "sdpa"
    }
}

bnb_float16_settings = {
    'tokenizer_settings': {
        'torch_dtype': torch.float16,
        # 'add_bos_token': False, # doublecheck this
    },
    'model_settings': {
        'torch_dtype': torch.float16,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        'low_cpu_mem_usage': True,
        # 'attn_implementation': "sdpa"
    }
}

@functools.lru_cache(maxsize=1)
def get_hf_token():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config.get('hf_access_token')
    return None


def check_if_model_is_gated(repo_id, hf_token):
    try:
        api = HfApi(token=False)
        repo_info = api.repo_info(repo_id, token=False)
        return getattr(repo_info, 'gated', False)
    except Exception:
        if hf_token:
            try:
                api_with_token = HfApi(token=hf_token)
                repo_info = api_with_token.repo_info(repo_id)
                return getattr(repo_info, 'gated', False)
            except Exception:
                return False
        return False


class _StopOnToken(StoppingCriteria):
    """Stop generation when any ID in `stop_ids` is produced."""
    def __init__(self, stop_ids):
        self.stop_ids = set(stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.stop_ids


class StopAfterThink(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        self.buffer += self.tokenizer.decode(input_ids[0, -1], skip_special_tokens=True)
        return "</think>" in self.buffer


class BaseModel(ABC):
    def __init__(self, model_info, settings, generation_settings, attn_implementation=None, tokenizer_kwargs=None, model_kwargs=None):
        if attn_implementation:
            settings = get_model_settings(settings, attn_implementation)
        self.model_info = model_info
        self.settings = settings
        self.model_name = model_info['model']
        self.generation_settings = generation_settings
        self.max_length = generation_settings['max_length']

        self.device = _configure_device_settings(self.settings, self.model_info)

        script_dir = Path(__file__).resolve().parent
        cache_dir = script_dir / "Models" / "chat" / self.model_info['cache_dir']

        hf_token = get_hf_token()
        
        is_gated = self.model_info.get('gated', False)
        if not is_gated:
            is_gated = check_if_model_is_gated(model_info['repo_id'], hf_token)

        tokenizer_settings = {
            **self.settings.get('tokenizer_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if tokenizer_kwargs:
            tokenizer_settings.update(tokenizer_kwargs)
        if is_gated and hf_token:
            tokenizer_settings['token'] = hf_token
        elif not is_gated:
            tokenizer_settings['token'] = False

        with utf8_file_operations():
            self.tokenizer = AutoTokenizer.from_pretrained(model_info['repo_id'], **tokenizer_settings)

        if tokenizer_kwargs and 'eos_token' in tokenizer_kwargs:
            self.tokenizer.eos_token = tokenizer_kwargs['eos_token']

        model_settings = {
            **self.settings.get('model_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if model_kwargs:
            model_settings.update(model_kwargs)

        if is_gated and hf_token:
            model_settings['token'] = hf_token
        elif not is_gated:
            model_settings['token'] = False

        self.model = AutoModelForCausalLM.from_pretrained(model_info['repo_id'], **model_settings)
        self.model.eval()

        config = self.model.config
        model_dtype = next(self.model.parameters()).dtype
        my_cprint(f"Loaded {model_info['model']} ({model_dtype}) on {self.device} using {config._attn_implementation}", "green")

    def get_model_name(self):
        return self.model_name

    @abstractmethod
    def create_prompt(self, augmented_query):
        pass

    def create_inputs(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

        if inputs['input_ids'].size(1) > self.max_length:
            raise ValueError(f"Input prompt is too long ({inputs['input_ids'].size(1)} tokens). Maximum length is {self.max_length} tokens.")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @torch.inference_mode()
    def generate_response(self, inputs, remove_token_type_ids=False):
        if remove_token_type_ids:
            inputs.pop('token_type_ids', None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        eos_token_id = self.tokenizer.eos_token_id

        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def switch_model(self, new_model_class):
        self.cleanup()
        return new_model_class()

    @staticmethod
    def free_torch_memory(model, tokenizer):
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class Granite(BaseModel):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]

        if '2b' in model_name.lower() and not torch.cuda.is_available():
            settings = {}
        else:
            settings = bnb_bfloat16_settings

        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|start_of_role|>system<|end_of_role|>{system_message}<|end_of_text|>
<|start_of_role|>user<|end_of_role|>{augmented_query}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""


class Exaone(BaseModel):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]

        settings = copy.deepcopy(bnb_bfloat16_settings)
        settings['tokenizer_settings']['trust_remote_code'] = True
        settings['model_settings']['trust_remote_code'] = True

        if '2.4b' in model_name.lower() and not torch.cuda.is_available():
            settings = {
                'tokenizer_settings': {'trust_remote_code': True},
                'model_settings': {'trust_remote_code': True}
            }

        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""[|system|]{system_message}[|endofturn|]
[|user|]{augmented_query}
[|assistant|]"""


class Qwen(BaseModel):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]
        
        is_small_model = (
            # '4b' in model_name.lower() or
            '1.7b' in model_name.lower() or 
            '0.6b' in model_name.lower()
        )
        no_cuda = not torch.cuda.is_available()
        
        if is_small_model and no_cuda:
            settings = {}
        else:
            settings = bnb_bfloat16_settings
            
        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""


class Mistral_Small_24b(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s>

[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT]

[INST]{augmented_query}[/INST]"""


class DeepseekR1(BaseModel):
    def __init__(self, generation_settings: dict, model_name: str):
        model_info = CHAT_MODELS[model_name]

        settings = deepcopy(bnb_bfloat16_settings)
        settings["tokenizer_settings"]["trust_remote_code"] = True
        settings["model_settings"]["trust_remote_code"] = True
        # settings["model_settings"]["attn_implementation"] = "sdpa"

        custom_generation_settings = {
            "max_length": generation_settings["max_length"],
            "max_new_tokens": generation_settings["max_new_tokens"],
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "num_beams": 1,
            "use_cache": True
        }

        tokenizer_kwargs = {
            "trust_remote_code": True,
        }

        super().__init__(
            model_info,
            settings,
            custom_generation_settings,
            attn_implementation=None,
            tokenizer_kwargs=tokenizer_kwargs
        )

        self.generation_settings["pad_token_id"] = self.tokenizer.eos_token_id

    def create_prompt(self, augmented_query: str) -> str:
        return f"""<｜begin_of_sentence｜>{system_message}<｜User｜>{augmented_query}<｜Assistant｜>"""

    @torch.inference_mode()
    def generate_response(self, inputs, remove_token_type_ids: bool = False):
        yield from super().generate_response(inputs, remove_token_type_ids)


class GLM4Z1(BaseModel):
    def __init__(self, generation_settings: dict, model_name: str):
        model_info = CHAT_MODELS[model_name]

        settings = deepcopy(bnb_bfloat16_settings)
        settings["tokenizer_settings"]["trust_remote_code"] = True
        settings["model_settings"]["trust_remote_code"] = True
        settings["model_settings"]["attn_implementation"] = "sdpa"

        custom_generation_settings = {
            "max_length": generation_settings["max_length"],
            "max_new_tokens": generation_settings["max_new_tokens"],
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "num_beams": 1,
            "use_cache": True
        }

        tokenizer_kwargs = {
            "trust_remote_code": True,
            "chat_template": GLM4Z1_CHAT_TEMPLATE
        }

        super().__init__(
            model_info,
            settings,
            custom_generation_settings,
            attn_implementation=None,
            tokenizer_kwargs=tokenizer_kwargs
        )

        self.generation_settings["pad_token_id"] = self.tokenizer.eos_token_id

    def create_prompt(self, augmented_query: str) -> str:
        return f"""[gMASK]<sop><|system|>
{system_message}<|user|>
{augmented_query}<|assistant|>
<think>"""

    @torch.inference_mode()
    def generate_response(self, inputs, remove_token_type_ids: bool = False):
        if remove_token_type_ids:
            inputs.pop("token_type_ids", None)

        settings = {**inputs, **self.generation_settings}
        generated = self.model.generate(**settings)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        idx = text.rfind("</think>") + len("</think>")
        yield text[idx:].strip()


class SeedCoder(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<[begin_of_sentence]>system
{system_message}

<[end_of_sentence]><[begin_of_sentence]>user
{augmented_query}<[begin_of_sentence]>assistant
"""

    @torch.inference_mode()
    def generate_response(self, inputs):
        """
        SeedCoder does not accept `token_type_ids`, so remove them
        before calling the parent generator.
        """
        inputs.pop("token_type_ids", None)
        yield from super().generate_response(inputs)


class Phi4(BaseModel):
    def __init__(self, generation_settings: dict, model_name: str):
        model_info = CHAT_MODELS[model_name]

        settings = copy.deepcopy(bnb_bfloat16_settings)
        settings["model_settings"]["attn_implementation"] = "sdpa"
        settings["model_settings"]["device_map"] = "auto"

        # Pure-CPU fallback: no quant-weights on GPU, force everything to CPU
        if not torch.cuda.is_available():
            settings = {"tokenizer_settings": {}, "model_settings": {"device_map": "cpu"}}

        super().__init__(model_info, settings, generation_settings)

        self.generation_settings["pad_token_id"] = self.tokenizer.eos_token_id

    def create_prompt(self, augmented_query: str) -> str:
        return (
            f"<|system|>{system_message}<|end|>"
            f"<|user|>{augmented_query}<|end|><|assistant|>"
        )

    @torch.inference_mode()
    def generate_response(self, inputs, remove_token_type_ids: bool = False):
        if remove_token_type_ids:
            inputs.pop("token_type_ids", None)

        eos_id   = self.tokenizer.eos_token_id
        user_id  = self.tokenizer.convert_tokens_to_ids("<|user|>")
        assist_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")

        stop_criteria = StoppingCriteriaList([_StopOnToken({user_id, eos_id})])

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False
        )

        gen_thread = threading.Thread(
            target=self.model.generate,
            kwargs={**inputs,
                    **self.generation_settings,
                    "streamer": streamer,
                    "eos_token_id": eos_id,
                    "pad_token_id": eos_id,
                    "stopping_criteria": stop_criteria},
            daemon=True
        )
        gen_thread.start()

        buffer, sent = "", 0
        ASSIST, USER, END = "<|assistant|>", "<|user|>", "<|end|>"

        for chunk in streamer:
            buffer += chunk

            if ASSIST in buffer:
                buffer = buffer.split(ASSIST)[-1]

            for tag in (USER, END):
                cut = buffer.find(tag)
                if cut != -1:
                    buffer = buffer[:cut]
                    streamer.break_on_eos = True

            clean = buffer.replace(ASSIST, "").replace(USER, "").replace(END, "")

            if len(clean) > sent:
                yield clean[sent:]
                sent = len(clean)

        gen_thread.join()


class Minicpm(Qwen):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]
        settings = copy.deepcopy(bnb_bfloat16_settings)
        settings['tokenizer_settings']['trust_remote_code'] = True
        settings['model_settings']['trust_remote_code'] = True

        is_small_model = ('0.5b' in model_name.lower())
        no_cuda = not torch.cuda.is_available()
        if is_small_model and no_cuda:
            settings = {
                'tokenizer_settings': {'trust_remote_code': True},
                'model_settings': {'trust_remote_code': True}
            }

        BaseModel.__init__(self, model_info, settings, generation_settings)

def generate_response(model_instance, augmented_query):
    prompt = model_instance.create_prompt(augmented_query)
    inputs = model_instance.create_inputs(prompt)
    for partial_response in model_instance.generate_response(inputs):
        yield partial_response

def choose_model(model_name):
    if model_name in CHAT_MODELS:
        model_class_name = CHAT_MODELS[model_name]['function']
        model_class = globals()[model_class_name]

        max_length = get_max_length(model_name)
        max_new_tokens = get_max_new_tokens(model_name)
        generation_settings = get_generation_settings(max_length, max_new_tokens)

        return model_class(generation_settings, model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")