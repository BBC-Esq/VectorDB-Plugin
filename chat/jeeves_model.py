import torch

import chat.base as module_chat
from chat.base import BaseModel, LiquidAI, Qwen
from core.constants import CHAT_MODELS, jeeves_system_message


def _unquantized_settings():
    settings = {"tokenizer_settings": {}, "model_settings": {}}
    if torch.cuda.is_available():
        settings["model_settings"]["attn_implementation"] = "sdpa"
    return settings


class JeevesLiquidAI(LiquidAI):
    def __init__(self, model_name):
        model_info = CHAT_MODELS[model_name]
        generation_settings = module_chat.get_generation_settings(
            module_chat.get_max_length(model_name),
            module_chat.get_max_new_tokens(model_name),
        )
        BaseModel.__init__(self, model_info, _unquantized_settings(), generation_settings)
        if torch.cuda.is_available():
            self.model.to("cuda")
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def create_prompt(self, augmented_query):
        return (
            "<|startoftext|><|im_start|>system\n"
            f"{jeeves_system_message}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{augmented_query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class JeevesQwen(Qwen):
    def __init__(self, model_name):
        model_info = CHAT_MODELS[model_name]
        generation_settings = module_chat.get_generation_settings(
            module_chat.get_max_length(model_name),
            module_chat.get_max_new_tokens(model_name),
        )
        BaseModel.__init__(self, model_info, _unquantized_settings(), generation_settings)
        if torch.cuda.is_available():
            self.model.to("cuda")
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def create_prompt(self, augmented_query):
        messages = [
            {"role": "system", "content": jeeves_system_message},
            {"role": "user", "content": augmented_query},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


_JEEVES_CLASSES = {
    "LiquidAI": JeevesLiquidAI,
    "Qwen": JeevesQwen,
}


_JEEVES_KEY_ALIASES = {
    "Qwen 3 - 0.6b": "Qwen 3 - 0.6b (Thinking)",
    "Qwen 3 - 1.7b": "Qwen 3 - 1.7b (Thinking)",
}


def load_jeeves_model(chat_model_key):
    chat_model_key = _JEEVES_KEY_ALIASES.get(chat_model_key, chat_model_key)
    info = CHAT_MODELS.get(chat_model_key)
    if info is None:
        raise ValueError(f"'{chat_model_key}' is not in CHAT_MODELS.")
    cls = _JEEVES_CLASSES.get(info.get("function"))
    if cls is None:
        raise ValueError(
            f"Ask Jeeves has no model class for family '{info.get('function')}' ({chat_model_key}). "
            f"Add one in chat/jeeves_model.py."
        )
    return cls(chat_model_key)
