import os
import traceback
import inspect
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
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
    Glm4vForConditionalGeneration,
    AutoModelForImageTextToText
)
from langchain_community.docstore.document import Document
from extract_metadata import extract_image_metadata
from utilities import my_cprint, has_bfloat16_support, set_cuda_paths
from constants import VISION_MODELS

set_cuda_paths()

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "models" / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PROMPT = (
    "Describe this image in as much detail as possible but do not repeat yourself. "
    "Your response should be no more than one paragraph, but the paragraph can be as long as you want."
)

def get_best_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def check_for_images(image_dir: Path) -> bool:
    """
    Return True if least one file whose suffix is in ALLOWED_EXTENSIONS. Uses os.listdir to avoid Windows pathlib race conditions.
    """
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
    # 1) Get model_config either from caller or from config.yaml
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

    # 2) Look up loader name from constants (single source of truth)
    if chosen_model not in VISION_MODELS:
        raise KeyError(f"Unknown vision model: {chosen_model}")
    loader_name = VISION_MODELS[chosen_model]['loader']

    # 3) Instantiate and run the correct loader
    loader_class = globals()[loader_name]
    loader = loader_class(model_config)

    image_dir = Path(__file__).parent / "Docs_for_DB"
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

    def initialize_model_and_tokenizer(self):
        raise NotImplementedError

    def process_images(self):
        image_dir = Path(__file__).parent / "Docs_for_DB"
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
                        extracted_metadata = extract_image_metadata(full_path)
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
        info = VISION_MODELS[chosen_model]
        cache_dir = CACHE_DIR / info["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.device == "cuda":
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            precision_str = "bfloat16" if use_bf16 else "float16"

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=[
                    "vision_model",
                    "language_model.model.norm", 
                    "language_model.output",
                    "language_model.model.rotary_emb",
                    "language_model.lm_head",
                    "mlp1"
                ]
            )
            model = AutoModel.from_pretrained(
                info['repo_id'],
                quantization_config=quant_config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=False
            ).eval()
            device_str = "CUDA"
        else:
            # CPU fallback
            dtype = torch.float32
            precision_str = "float32"
            model = AutoModel.from_pretrained(
                info['repo_id'],
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=False,
                device_map={"": "cpu"}
            ).eval()
            device_str = "CPU"

        self.model_dtype = dtype
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")

        tokenizer = AutoTokenizer.from_pretrained(
            info['repo_id'],
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )

        return model, tokenizer, None

    def find_closest_aspect_ratio(self, aspect_ratio, ratios, w, h, sz):
        best_diff = float('inf')
        best = (1, 1)
        area = w * h
        for r in ratios:
            ar = r[0] / r[1]
            diff = abs(aspect_ratio - ar)
            if diff < best_diff or (diff == best_diff and area > 0.5 * sz * sz * r[0] * r[1]):
                best_diff = diff
                best = r

        return best

    def _build_transform(self, size):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((size, size), interpolation=InterpolationMode.LANCZOS, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def dynamic_preprocess(self, img, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        w, h = img.size
        ar = w / h
        ratios = sorted(
            {(i, j)
             for n in range(min_num, max_num + 1)
             for i in range(1, n + 1)
             for j in range(1, n + 1)
             if i * j <= max_num and i * j >= min_num},
            key=lambda x: x[0] * x[1]
        )
        best = self.find_closest_aspect_ratio(ar, ratios, w, h, image_size)
        tw, th = image_size * best[0], image_size * best[1]
        resized = img.resize((tw, th))
        blocks = best[0] * best[1]
        cols = tw // image_size
        parts = []
        for i in range(blocks):
            x = (i % cols) * image_size
            y = (i // cols) * image_size
            parts.append(resized.crop((x, y, x + image_size, y + image_size)))
        if use_thumbnail and len(parts) != 1:
            parts.append(img.resize((image_size, image_size)))

        return parts

    def _prepare_image(self, raw_image, input_size=448, max_num=24):
        imgs = self.dynamic_preprocess(raw_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tf = self._build_transform(input_size)

        return torch.stack([tf(im) for im in imgs])

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        pv = self._prepare_image(raw_image).to(self.model_dtype).to(self.device)

        question = f"<image>\n{IMAGE_PROMPT}"

        gen_cfg = {
            'num_beams': 1,
            'max_new_tokens': 512,
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        resp = self.model.chat(self.tokenizer, pv, question, gen_cfg)

        return ' '.join(line.strip() for line in resp.split('\n') if line.strip())


class loader_granite(BaseLoader):
    """
    Loader for Granite Vision (3.2-2B) with runtime overrides for `image_grid_pinpoints` to control VRAM usage.
    """

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

        # 1) Low
        low_tiling_pinpoints = [[384, 384], [768, 384], [384, 768]]

        # 2) Medium
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

        # High
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

        # 3) All (default from model JSON)
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

        # Pick a tiling pinpoint preset
        # custom_pinpoints = low_tiling_pinpoints
        custom_pinpoints = medium_tiling_pinpoints
        # custom_pinpoints = high_tiling_pinpoints
        # custom_pinpoints = all_tiling_pinpoints

        # Use custom_pinpoints during preprocessing
        try:
            processor.image_grid_pinpoints = custom_pinpoints
        except Exception:
            pass

        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "image_grid_pinpoints"):
            ip.image_grid_pinpoints = custom_pinpoints

        if self.device == "cuda" and torch.cuda.is_available():
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            precision_str = "bfloat16" if use_bf16 else "float16"

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
            # CPU mode - no quantization
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map={"": "cpu"}
            )
            my_cprint(f"{chosen_model} loaded into memory on CPU (float32)", "green")

        # Use custom_pinpoints during inference
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
        return " ".join(line.strip() for line in resp.split("\n") if line.strip())


class loader_qwenvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info['cache_dir']
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16

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
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )
        model = model.to(self.device)
        model.eval()

        precision_str = "bfloat16" if use_bf16 else "float16"
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

        return ' '.join(line.strip() for line in response.split('\n') if line.strip())


class loader_glmv4_thinking(BaseLoader):
    """
    Loader for GLM-4v-thinking with a **pre-resize** pixel-budget cap to control VRAM.
    We pre-resize the PIL image BEFORE passing it to the processor because the comments
    within the transformers source code falsely states that longest_edge is exposed.
    """

    PIXELS_LOW     = 294_912 # ≈ 384×768
    PIXELS_MEDIUM  = 589_824 # ≈ 768×768 or 384×1536
    PIXELS_HIGH    = 1_179_648 # ≈ 768×1536 or 384×3072
    PIXELS_DEFAULT = 4_816_896  # mirrors library default behavior

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        model_id = info['repo_id']
        save_dir = info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        processor = AutoProcessor.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)

        # CHOOSE PIXEL BUDGET TIER (adjust here)
        # self.pixel_cap = self.PIXELS_LOW
        # self.pixel_cap = self.PIXELS_MEDIUM
        self.pixel_cap = self.PIXELS_HIGH
        # self.pixel_cap = self.PIXELS_DEFAULT

        model = Glm4vForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            quantization_config=quant_config,
            cache_dir=cache_dir,
        ).eval()

        self.device = torch.device("cuda")
        self.model = model
        self.processor = processor

        my_cprint(f"{chosen_model} loaded into memory on CUDA (bfloat16)", "green")
        return model, None, processor

    def _cap_pixels_for_glm4v(self, pil_img, max_pixels_2d, divisor=28):

        w, h = pil_img.size
        area = w * h

        # If already within budget, snap to multiples of divisor without enlarging.
        if area <= max_pixels_2d:
            new_w = max(divisor, (w // divisor) * divisor)
            new_h = max(divisor, (h // divisor) * divisor)
            if new_w == w and new_h == h:
                return pil_img
            return pil_img.resize((new_w, new_h), Image.BICUBIC)

        # Otherwise, scale by sqrt so area shrinks to the target budget, then snap to divisor.
        scale = (max_pixels_2d / float(area)) ** 0.5
        new_w = max(divisor, int((w * scale) // divisor * divisor))
        new_h = max(divisor, int((h * scale) // divisor * divisor))
        if new_w < divisor or new_h < divisor:
            new_w = new_h = divisor
        return pil_img.resize((new_w, new_h), Image.BICUBIC)

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        ip = getattr(self.processor, "image_processor", None)
        patch_size = getattr(ip, "patch_size", 14)
        merge_size = getattr(ip, "merge_size", 2)
        divisor = patch_size * merge_size

        raw_image = self._cap_pixels_for_glm4v(
            raw_image,
            max_pixels_2d=self.pixel_cap,
            divisor=divisor,
        )

        prompt = (
            "[gMASK]<sop><|user|>\n"
            "<|begin_of_image|><|image|><|end_of_image|>"
            f"{IMAGE_PROMPT}"
            "<|assistant|>\n"
        )

        inputs = self.processor(
            text=prompt,
            images=raw_image,
            return_tensors="pt",
        ).to("cuda")

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
        torch.cuda.synchronize()

        generated_ids_trimmed = [out_ids[0][len(inputs.input_ids[0]):]]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if '<answer>' in response and '</answer>' in response:
            start_idx = response.find('<answer>') + len('<answer>')
            end_idx = response.find('</answer>')
            response = response[start_idx:end_idx].strip()

        return response


class loader_liquidvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        source = info.get('model_path') or info['repo_id']
        cache_dir = CACHE_DIR / info.get('cache_dir', '')
        cache_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            precision_str = "bfloat16" if use_bf16 else "float16"
            device_map = "auto"
        else:
            dtype = torch.float32
            precision_str = "float32"
            device_map = {"": "cpu"}

        model = AutoModelForImageTextToText.from_pretrained(
            source,
            trust_remote_code=True,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            device_map=device_map,
        ).eval()

        processor = AutoProcessor.from_pretrained(
            source,
            trust_remote_code=True,
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
        return " ".join(line.strip() for line in text.split("\n") if line.strip())
