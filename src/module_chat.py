import yaml
import logging
import gc
import copy
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import threading
from abc import ABC, abstractmethod

from constants import CHAT_MODELS, system_message, MODEL_MAX_TOKENS, MODEL_MAX_NEW_TOKENS
from utilities import my_cprint, has_bfloat16_support

# logging.getLogger("transformers").setLevel(logging.WARNING) # adjust to see deprecation and other non-fatal errors
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_model_settings(base_settings, attn_implementation):
    settings = copy.deepcopy(base_settings)
    # settings['model_settings']['attn_implementation'] = attn_implementation
    return settings
    
def get_max_length(model_name):
    return MODEL_MAX_TOKENS.get(model_name, 8192)

def get_max_new_tokens(model_name):
    return MODEL_MAX_NEW_TOKENS.get(model_name, 1024)

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

def get_hf_token():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config.get('hf_access_token')
    return None

class BaseModel(ABC):
    def __init__(self, model_info, settings, generation_settings, attn_implementation=None, tokenizer_kwargs=None, model_kwargs=None):
        if attn_implementation:
            settings = get_model_settings(settings, attn_implementation)
        self.model_info = model_info
        self.settings = settings
        self.model_name = model_info['model']
        self.generation_settings = generation_settings
        self.max_length = generation_settings['max_length']
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        script_dir = Path(__file__).resolve().parent
        cache_dir = script_dir / "Models" / "chat" / model_info['cache_dir']

        # rewrite bfloat dictionary to float16 if bfloat16 not supported
        if self.device == "cuda" and not has_bfloat16_support():
            if 'bnb_bfloat16_settings' in settings:
                settings['bnb_float16_settings'] = settings.pop('bnb_bfloat16_settings')
                settings['bnb_float16_settings']['tokenizer_settings']['torch_dtype'] = torch.float16
                settings['bnb_float16_settings']['model_settings']['torch_dtype'] = torch.float16
                settings['bnb_float16_settings']['model_settings']['quantization_config'].bnb_4bit_compute_dtype = torch.float16

        hf_token = get_hf_token()

        tokenizer_settings = {
            **settings.get('tokenizer_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if tokenizer_kwargs:
            tokenizer_settings.update(tokenizer_kwargs)
        if hf_token:
            tokenizer_settings['use_auth_token'] = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_info['repo_id'], **tokenizer_settings)

        if tokenizer_kwargs and 'eos_token' in tokenizer_kwargs:
            self.tokenizer.eos_token = tokenizer_kwargs['eos_token']

        model_settings = {
            **settings.get('model_settings', {}), 
            'cache_dir': str(cache_dir)
        }
        if model_kwargs:
            model_settings.update(model_kwargs)

        # if using CPU, remove CUDA-specific settings
        # only applies to Zephyr 1.6b because all other models are not populated in combobox if cuda isn't available
        if self.device == "cpu":
            model_settings.pop('quantization_config', None)
            # model_settings.pop('attn_implementation', None)
            model_settings['device_map'] = "cpu"

        if hf_token:
            model_settings['use_auth_token'] = hf_token

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def generate_response(self, inputs, remove_token_type_ids=False):
        """
        Creates a TextIteratorStreamer to stream partial responses.
        """
        if remove_token_type_ids:
            inputs.pop('token_type_ids', None)
            
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        eos_token_id = self.tokenizer.eos_token_id
        
        all_settings = {**inputs, **self.generation_settings, 'streamer': streamer, 'eos_token_id': eos_token_id}

        # generation + streamer require two threads to work
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

    def cleanup_resources(model, tokenizer):
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class Zephyr(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]

        if '1.6b' in model_name.lower():
            settings = bnb_float16_settings if torch.cuda.is_available() else {}
        else:
            settings = bnb_bfloat16_settings

        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|system|>
{system_message}<|endoftext|>
<|user|>
{augmented_query}<|endoftext|>
<|assistant|>
"""


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

        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""[|system|]{system_message}[|endofturn|]
[|user|]{augmented_query}
[|endofturn|]
[|assistant|]"""


class Qwen(BaseModel):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]

        if '1.5b' in model_name.lower() and not torch.cuda.is_available():
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


class QwenCoder(BaseModel):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]
        
        if '1.5b' in model_name.lower() and not torch.cuda.is_available():
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

    def generate_response(self, inputs):
        return super().generate_response(inputs, remove_token_type_ids=True)


class DeepseekR1(BaseModel):
    def __init__(self, generation_settings, model_name):
        model_info = CHAT_MODELS[model_name]
        
        custom_generation_settings = {
            'max_length': generation_settings['max_length'],
            'max_new_tokens': generation_settings['max_new_tokens'],
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.95,
            'use_cache': True,
            'num_beams': 1
        }

        settings = bnb_bfloat16_settings

        super().__init__(model_info, settings, custom_generation_settings)
        
    def create_prompt(self, augmented_query):
        return f"""<|begin_of_sentence|>{system_message}<|User|>{augmented_query}<|Assistant|><｜end_of_sentence｜><｜Assistant｜>"""

    def generate_response(self, inputs):
        SHOW_THINKING = False
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        all_settings = {
            **inputs, 
            **self.generation_settings,
            'streamer': streamer, 
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        if SHOW_THINKING:
            # stream everything
            for partial_response in streamer:
                yield partial_response
        else:
            # only stream after </think> tag
            buffer = ""
            thinking_complete = False

            for partial_response in streamer:
                buffer += partial_response
                if not thinking_complete and '</think>' in buffer:
                    thinking_complete = True
                    start_idx = buffer.rfind('</think>') + len('</think>')
                    yield buffer[start_idx:].strip()
                    buffer = ""
                elif thinking_complete:
                    yield partial_response

        generation_thread.join()


class Internlm3(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]
        settings = copy.deepcopy(bnb_bfloat16_settings)
        settings['tokenizer_settings']['trust_remote_code'] = True
        settings['model_settings']['trust_remote_code'] = True
        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|begin_of_text|><|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
       inputs.pop('token_type_ids', None)

       streamer = TextIteratorStreamer(
           self.tokenizer, 
           skip_prompt=True, 
           skip_special_tokens=True
       )
       
       eos_token_id = self.tokenizer.convert_tokens_to_ids(['<|im_end|>'])[0]

       all_settings = {
           **inputs, 
           **self.generation_settings, 
           'streamer': streamer, 
           'eos_token_id': eos_token_id,
           'pad_token_id': self.tokenizer.pad_token_id
       }

       generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
       generation_thread.start()

       for partial_response in streamer:
           yield partial_response

       generation_thread.join()


class OLMo2_13b(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]
        settings = bnb_bfloat16_settings if torch.cuda.is_available() else {}
        super().__init__(model_info, settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|endoftext|><|system|>
{system_message}
<|user|>
{augmented_query}
<|assistant|>
"""

    def generate_response(self, inputs):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        all_settings = {
            **inputs, 
            **self.generation_settings, 
            'streamer': streamer, 
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        for partial_response in streamer:
            yield partial_response

        generation_thread.join()


class Mistral_Small_24b(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]
        super().__init__(model_info, bnb_bfloat16_settings, generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<s>

[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT]

[INST]{augmented_query}[/INST]"""


class QwQ_32b(BaseModel):
    def __init__(self, generation_settings, model_name=None):
        model_info = CHAT_MODELS[model_name]

        custom_generation_settings = {
            'max_length': generation_settings['max_length'],
            'max_new_tokens': generation_settings['max_new_tokens'],
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.95,
            'use_cache': True,
            'num_beams': 1
        }

        super().__init__(model_info, bnb_bfloat16_settings, custom_generation_settings)

    def create_prompt(self, augmented_query):
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{augmented_query}<|im_end|>
<|im_start|>assistant
"""

    def generate_response(self, inputs):
        SHOW_THINKING = False
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        all_settings = {
            **inputs, 
            **self.generation_settings,
            'streamer': streamer, 
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }

        generation_thread = threading.Thread(target=self.model.generate, kwargs=all_settings)
        generation_thread.start()

        if SHOW_THINKING:
            # stream everything
            for partial_response in streamer:
                yield partial_response
        else:
            # only stream after </think> tag
            buffer = ""
            thinking_complete = False

            for partial_response in streamer:
                buffer += partial_response
                if not thinking_complete and '</think>' in buffer:
                    thinking_complete = True
                    start_idx = buffer.rfind('</think>') + len('</think>')
                    yield buffer[start_idx:].strip()
                    buffer = ""
                elif thinking_complete:
                    yield partial_response

        generation_thread.join()


@torch.inference_mode()
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