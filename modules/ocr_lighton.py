import html
import logging
import math
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_BLOCKTIME", "0")

import torch
from PIL import Image

from core.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODEL_NAME = "LightOnOCR-3-0.8B"
MODEL_REPO = "lightonai/LightOnOCR-3-0.8B"
MODEL_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
MODEL_FILES = ["*.json", "*.jinja", "*.txt", "*.safetensors"]
MAX_PIXELS = 5_000_000
MAX_DPI = 400
MAX_NEW_TOKENS = 12288
TEMPERATURE = 0.2
TOP_P = 0.9
SEED = 0
MINIMUM_VRAM_GB = 5.5
BATCH_BY_VRAM_GB = ((15.5, 10), (11.5, 6), (7.5, 2), (MINIMUM_VRAM_GB, 1))
VRAM_BASE_GB = 2.2
VRAM_PER_PAGE_GB = 0.76
VRAM_WEIGHTS_GB = 1.7
VRAM_MARGIN_GB = 1.0


def model_dir():
    return Path(PROJECT_ROOT) / "Models" / "ocr" / MODEL_NAME


def total_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def batch_for_card(total_gb):
    for minimum, batch in BATCH_BY_VRAM_GB:
        if total_gb >= minimum:
            return batch
    return 0


def available():
    return batch_for_card(total_vram_gb()) > 0


def ensure_model():
    folder = model_dir()
    if (folder / "model.safetensors").exists() and (folder / "config.json").exists():
        return folder
    from huggingface_hub import snapshot_download
    print(f"Downloading the OCR model {MODEL_NAME} (about 1.7 GB)", flush=True)
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=str(folder), allow_patterns=MODEL_FILES)
    return folder


def load():
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    folder = str(ensure_model())
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(folder)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        folder, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    return model, processor


def batch_size(pages):
    size = max(1, min(batch_for_card(total_vram_gb()), pages))
    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
    while size > 1 and VRAM_BASE_GB + VRAM_PER_PAGE_GB * size - VRAM_WEIGHTS_GB + VRAM_MARGIN_GB > free_gb:
        size -= 1
    return size


def render(page):
    area = (page.rect.width / 72) * (page.rect.height / 72)
    dpi = max(36, min(MAX_DPI, math.floor(math.sqrt(MAX_PIXELS / max(area, 0.01)))))
    pixmap = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    if image.width * image.height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (image.width * image.height))
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    return image


def _end_tokens(model):
    end = model.generation_config.eos_token_id
    if end is None:
        return set()
    return set(end if isinstance(end, (list, tuple)) else [end])


def read_images(model, processor, images):
    conversations = [[{"role": "user", "content": [{"type": "image", "image": image}]}] for image in images]
    inputs = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", processor_kwargs={"padding": True})
    device = next(model.parameters()).device
    inputs = {key: (value.to(device, model.dtype) if value.is_floating_point() else value.to(device))
              for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    options = {"max_new_tokens": MAX_NEW_TOKENS, "use_cache": True, "do_sample": True,
               "temperature": TEMPERATURE, "top_p": TOP_P}
    padding = model.generation_config.pad_token_id
    if padding is None:
        padding = processor.tokenizer.pad_token_id
    if padding is not None:
        options["pad_token_id"] = padding
    torch.manual_seed(SEED)
    with torch.no_grad():
        output = model.generate(**inputs, **options)
    end_tokens = _end_tokens(model)
    results = []
    for row in range(len(images)):
        tokens = output[row, prompt_length:].tolist()
        end = next((i for i, token in enumerate(tokens) if token in end_tokens), None)
        cut_off = end is None and len(tokens) >= MAX_NEW_TOKENS
        if end is not None:
            tokens = tokens[:end + 1]
        results.append((to_plain(processor.decode(tokens, skip_special_tokens=True)), cut_off))
    del output, inputs
    return results


def read_in_batches(model, processor, items, render_item, on_result):
    todo = list(items)
    size = batch_size(len(todo))
    while todo:
        group, todo = todo[:size], todo[size:]
        try:
            results = read_images(model, processor, [render_item(item) for item in group])
        except torch.OutOfMemoryError:
            if len(group) == 1:
                raise
            torch.cuda.empty_cache()
            size = max(1, len(group) // 2)
            todo = group + todo
            continue
        for item, (text, cut_off) in zip(group, results):
            on_result(item, text, cut_off)


_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*(?:\s[^<>\n]*)?/?>")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_MODEL_TOKEN = re.compile(r"<\|[^|<>\n]{0,40}\|>")
_BROKEN_TAG = re.compile(r"</\s+[A-Za-z][A-Za-z0-9]*\s*>|</?(?:table|thead|tbody|tfoot|tr|td|th|div|span|br|"
                         r"p|b|i|u|sup|sub|li|ul|ol|img|em|strong|h[1-6])(?=[\s<]|$)")


def _table_row(match):
    cells = [re.sub(r"\s+", " ", _TAG.sub("", cell)).strip() for cell in re.split(r"(?i)</t[dh]\s*>", match.group(1))]
    while cells and not cells[-1]:
        cells.pop()
    return "\t".join(cells) + "\n"


def to_plain(text):
    text = _MARKDOWN_IMAGE.sub(" ", text or "")
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = re.sub(r"(?is)<tr\b[^>]*>(.*?)</tr\s*>", _table_row, text)
    text = re.sub(r"(?i)</t[dh]\s*>", "\t", text)
    text = re.sub(r"(?i)</tr\s*>|<br\s*/?>|</p\s*>|</li\s*>|</h[1-6]\s*>|</div\s*>", "\n", text)
    text = _MODEL_TOKEN.sub(" ", text)
    text = _TAG.sub("", text)
    text = _BROKEN_TAG.sub(" ", text)
    text = html.unescape(text)
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        if re.fullmatch(r"\s*```[A-Za-z0-9_+-]*\s*", line) or re.fullmatch(r"\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        if line.lstrip().startswith("|"):
            line = "\t".join(cell.strip() for cell in line.strip().strip("|").split("|"))
        line = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", line)
        line = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"\1", line)
        lines.append(line.strip(" ").rstrip("\t"))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    text = re.sub(r"(\t[^\n]*)\n\n(?=[^\n]*\t)", r"\1\n", text)
    return text.strip()
