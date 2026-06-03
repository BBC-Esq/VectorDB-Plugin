import os
import traceback
import inspect
import time
import types
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    GenerationConfig,
    AutoConfig,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText
)
from db.document_processor import Document
from core.extract_metadata import extract_typed_metadata
from core.utilities import my_cprint, has_bfloat16_support, set_cuda_paths
from core.constants import VISION_MODELS, PROJECT_ROOT

set_cuda_paths()

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

current_directory = PROJECT_ROOT
CACHE_DIR = current_directory / "models" / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PROMPT = (
    "Describe this image in as much detail as possible but do not repeat yourself. "
    "Your response should be no more than one paragraph, but the paragraph can be as long as you want."
)

def get_best_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def check_for_images(image_dir: Path) -> bool:
    try:
        filenames = os.listdir(str(image_dir))
        return any(Path(f).suffix.lower() in ALLOWED_EXTENSIONS for f in filenames)
    except FileNotFoundError:
        return False
    except OSError:
        return False

def run_loader_in_process(loader_func):
    try:
        return loader_func()
    except Exception as e:
        error_message = f"Error processing images: {e}\n\nTraceback:\n{traceback.format_exc()}"
        my_cprint(error_message, "red")
        return []


def choose_image_loader(model_config: dict | None = None):
    if model_config is None:
        cfg_path = Path('config.yaml')
        if not cfg_path.exists():
            raise FileNotFoundError("config.yaml not found and no model_config provided")
        with cfg_path.open('r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f) or {}

    vision_cfg = (model_config.get('vision') or {})
    chosen_model = vision_cfg.get('chosen_model')
    if not chosen_model:
        raise ValueError("vision.chosen_model missing in config/model_config")

    if chosen_model not in VISION_MODELS:
        raise KeyError(f"Unknown vision model: {chosen_model}")
    loader_name = VISION_MODELS[chosen_model]['loader']

    loader_class = globals()[loader_name]
    loader = loader_class(model_config)

    image_dir = PROJECT_ROOT / "Docs_for_DB"
    if not check_for_images(image_dir):
        return []

    with ProcessPoolExecutor(1, initializer=set_cuda_paths) as executor:
        future = executor.submit(run_loader_in_process, loader.process_images)
        try:
            processed_docs = future.result()
        except Exception as e:
            my_cprint(f"Error occurred during image processing: {e}", "red")
            return []
        return processed_docs or []


class BaseLoader:
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    @staticmethod
    def detect_dtype():
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        return (torch.bfloat16, "bfloat16") if use_bf16 else (torch.float16, "float16")

    @staticmethod
    def normalize_response(text):
        return ' '.join(line.strip() for line in text.split('\n') if line.strip())

    def initialize_model_and_tokenizer(self):
        raise NotImplementedError

    def process_images(self):
        image_dir = PROJECT_ROOT / "Docs_for_DB"
        documents = []

        try:
            image_files = [file for file in image_dir.iterdir() if file.suffix.lower() in ALLOWED_EXTENSIONS]
        except OSError:
            image_files = []
            print(f"Error accessing directory {image_dir}")

        self.model, self.tokenizer, self.processor = self.initialize_model_and_tokenizer()
        print("Processing images.")
        start_time = time.time()
        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for full_path in image_files:
                try:
                    with Image.open(full_path) as raw_image:
                        extracted_text = self.process_single_image(raw_image)
                        extracted_metadata = extract_typed_metadata(full_path, "image")
                        documents.append(Document(page_content=extracted_text, metadata=extracted_metadata))
                        progress_bar.update(1)
                except Exception as e:
                    print(f"{full_path.name}: Error processing image - {e}")
        total_time = time.time() - start_time
        print(f"Loaded {len(documents)} image(s).")
        print(f"Total image processing time: {total_time:.2f} seconds")
        my_cprint("Vision model removed from memory.", "red")
        return documents

    def process_single_image(self, raw_image):
        raise NotImplementedError


class loader_internvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        cache_dir = CACHE_DIR / model_info["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        dtype, precision_str = self.detect_dtype()

        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=cache_dir,
            token=False,
        )

        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
            )
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map="auto",
            )
            device_str = "CUDA"
        else:
            dtype = torch.float32
            precision_str = "float32"
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map={"": "cpu"},
            )
            device_str = "CPU"

        model.eval()
        self.model_dtype = dtype
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")

        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.model_dtype)

        input_len = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        new_tokens = output[:, input_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        return self.normalize_response(text)


class loader_granite(BaseLoader):

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_id = VISION_MODELS[chosen_model]['repo_id']
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=cache_dir,
            token=False
        )

        low_tiling_pinpoints = [[384, 384], [768, 384], [384, 768]]

        medium_tiling_pinpoints = [
            [384, 384],
            [384, 768],
            [768, 384],
            [384, 1152],
            [1152, 384],
            [384, 1536],
            [768, 768],
            [1536, 384],
        ]

        high_tiling_pinpoints = [
            [384, 384],
            [384, 768],
            [768, 384],
            [384, 1152],
            [1152, 384],
            [384, 1536],
            [768, 768],
            [1536, 384],
            [384, 1920],
            [1920, 384],
            [384, 2304],
            [768, 1152],
            [1152, 768],
            [2304, 384],
        ]

        all_tiling_pinpoints = [
            [384, 384], [384, 768], [384, 1152], [384, 1536],
            [384, 1920], [384, 2304], [384, 2688], [384, 3072],
            [384, 3456], [384, 3840],
            [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920],
            [1152, 384], [1152, 768], [1152, 1152],
            [1536, 384], [1536, 768],
            [1920, 384], [1920, 768],
            [2304, 384], [2688, 384], [3072, 384], [3456, 384], [3840, 384]
        ]

        custom_pinpoints = medium_tiling_pinpoints

        try:
            processor.image_grid_pinpoints = custom_pinpoints
        except Exception:
            pass

        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "image_grid_pinpoints"):
            ip.image_grid_pinpoints = custom_pinpoints

        if self.device == "cuda" and torch.cuda.is_available():
            dtype, precision_str = self.detect_dtype()

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=[
                    "vision_tower",
                    "multi_modal_projector",
                    "language_model.embed_tokens",
                    "language_model.norm",
                    "lm_head"
                ]
            )

            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                quantization_config=quant_cfg,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map="auto"
            )
            my_cprint(f"{chosen_model} loaded into memory on CUDA ({precision_str})", "green")

        else:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map={"": "cpu"}
            )
            my_cprint(f"{chosen_model} loaded into memory on CPU (float32)", "green")

        try:
            if hasattr(model, "config") and hasattr(model.config, "image_grid_pinpoints"):
                model.config.image_grid_pinpoints = custom_pinpoints
        except Exception:
            pass
        if hasattr(model, "image_grid_pinpoints"):
            try:
                setattr(model, "image_grid_pinpoints", custom_pinpoints)
            except Exception:
                pass

        model.eval()

        self.model = model
        self.processor = processor

        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        prompt = f"<|user|>\n<image>\n{IMAGE_PROMPT}\n<|assistant|>\n"

        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1
        )

        resp = self.processor.decode(output[0], skip_special_tokens=True).split('<|assistant|>')[-1].strip()
        return self.normalize_response(resp)


class loader_qwenvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info['cache_dir']
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        dtype, _ = self.detect_dtype()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=[
                "lm_head",
                "merger",
                "visual.blocks.0.attn",
                "visual.blocks.0.mlp",
                "visual.blocks.1.attn",
                "visual.blocks.1.mlp",
                "visual.blocks.2.attn",
                "visual.blocks.2.mlp",
                "visual.blocks.3.attn",
                "visual.blocks.3.mlp",
                "visual.blocks.4.attn",
                "visual.blocks.5.mlp",
                "visual.blocks.7.attn",
                "visual.blocks.7.mlp",
                "visual.blocks.8.mlp",
                "visual.blocks.10.mlp",
                "visual.blocks.12.mlp",
                "visual.blocks.13.mlp",
                "visual.blocks.14.attn",
                "visual.blocks.14.mlp",
                "visual.blocks.15.attn",
                "visual.blocks.15.mlp",
                "visual.blocks.17.mlp",
                "visual.blocks.31.mlp.down_proj"
            ]
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            min_pixels=28*28,
            max_pixels=1280*28*28,
            cache_dir=cache_dir,
            token=False
        )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            token=False,
            device_map="auto",
        )
        model.eval()
        self._replace_conv3d_patch_embed_with_matmul(model)

        _, precision_str = self.detect_dtype()
        device_str = "CUDA" if self.device == "cuda" else "CPU"
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")

        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):

        prompt = (
            "<|im_start|>user\n"
            f"{IMAGE_PROMPT} <|vis_start|><|image_pad|><|vis_end|>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            top_k=None,
            top_p=None,
            num_beams=1,
            temperature=None
        )
        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = response.split('assistant')[-1].strip()

        return self.normalize_response(response)

    def _replace_conv3d_patch_embed_with_matmul(self, model):
        """Replace the Qwen-VL vision patch-embed Conv3d with its exact matmul equivalent.

        The patch embed is an ``nn.Conv3d``. On some GPUs cuDNN has no fast bf16/fp16
        3D-convolution kernel for certain shapes and falls back to a path ~100x slower:
        Qwen3-VL's conv took ~12-15s per image here (the entire prefill), while
        Qwen2.5-VL's differently shaped conv hits a fast kernel and stays ~0.15s -- same
        code and dtypes, only the conv dimensions differ.

        Because ``kernel == stride == patch size``, each patch is an independent linear
        projection, so the conv is mathematically an exact matmul. Swapping it for that
        matmul sidesteps cuDNN: microseconds in bf16, numerically identical to within
        bf16 rounding. No-op if the patch embed is not an ``nn.Conv3d``.
        """
        try:
            patch_embed = model.model.visual.patch_embed
            proj = patch_embed.proj
            if not isinstance(proj, torch.nn.Conv3d):
                return
            weight = proj.weight.detach().reshape(proj.weight.shape[0], -1).t().contiguous()
            bias = proj.bias.detach() if proj.bias is not None else None
            in_features = weight.shape[0]

            def forward(self, hidden_states):
                hidden_states = hidden_states.view(-1, in_features).to(weight.dtype)
                if bias is None:
                    return hidden_states.matmul(weight)
                return torch.addmm(bias, hidden_states, weight)

            patch_embed.forward = types.MethodType(forward, patch_embed)
        except Exception as exc:
            my_cprint(f"Qwen-VL patch-embed optimization skipped: {exc}", "yellow")


class loader_liquidvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        source = info.get('model_path') or info['repo_id']
        cache_dir = CACHE_DIR / info.get('cache_dir', '')
        cache_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            dtype, precision_str = self.detect_dtype()
            device_map = "auto"
        else:
            dtype = torch.float32
            precision_str = "float32"
            device_map = {"": "cpu"}

        model = AutoModelForImageTextToText.from_pretrained(
            source,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            device_map=device_map,
        ).eval()

        processor = AutoProcessor.from_pretrained(
            source,
            cache_dir=cache_dir,
        )

        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "add_bos_token"):
            processor.tokenizer.add_bos_token = False

        if torch.cuda.is_available():
            device_str = "CUDA"
        else:
            device_str = "CPU"
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")

        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        system_text = "You are a helpful multimodal assistant."

        chatml = (
            "<|startoftext|><|im_start|>system\n"
            f"{system_text}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<image>{IMAGE_PROMPT}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = self.processor(
            text=[chatml],
            images=[[raw_image]],
            return_tensors="pt",
            use_image_special_tokens=True,
            do_image_splitting=True,
            min_image_tokens=64,
            max_image_tokens=256,
        )

        move_to = getattr(self.model, "device", None)
        if move_to is not None:
            inputs = inputs.to(move_to)

        input_len = inputs["input_ids"].shape[1]
        eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        pad_id = self.processor.tokenizer.pad_token_id or eos_id

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

        new_tokens = outputs[:, input_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return self.normalize_response(text)
