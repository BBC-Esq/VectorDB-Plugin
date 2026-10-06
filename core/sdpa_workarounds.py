import torch
import transformers.integrations.sdpa_attention as sdpa_attention

# TEMPORARY WORKAROUND - remove after upgrading torch to 2.14.0 or newer, which fixes the bug below
# (pytorch/pytorch#191937, fixed by #191984 and #192138; transformers declined to add a workaround of its own).
# PyTorch's memory-efficient SDPA kernel returns a wrong output for the last query position when K and V are
# stride-0 expanded views and an explicit causal mask is passed, if the key length is one more than a multiple
# of its key block (33, 65, 97, ... for head_dim 256; 65, 129, ... for head_dim 64/128). transformers' repeat_kv
# produces exactly those views for single-KV-head models whenever a padded batch carries an attention mask, so
# Harrier-270m got corrupted vectors for the full-length rows of such batches. Copying K/V to contiguous memory
# avoids the bug and leaves every other model bit-identical.
_stock_repeat_kv = sdpa_attention.repeat_kv


def _repeat_kv_contiguous(hidden_states, n_rep):
    repeated = _stock_repeat_kv(hidden_states, n_rep)
    return repeated.contiguous() if n_rep > 1 else repeated


sdpa_attention.repeat_kv = _repeat_kv_contiguous

# TEMPORARY WORKAROUND - remove once Windows builds of torch include FlashAttention or torch's memory-efficient SDPA
# kernel supports grouped-query attention (enable_gqa). For a batch with no padding, and for every single-text query,
# transformers drops the attention mask and passes enable_gqa=True instead of repeating the key/value heads. Without
# FlashAttention, SDPA then falls back to its math kernel, which builds the full attention matrix (about 4x slower and
# 7x the VRAM on 2,048-token chunks). Repeating the key/value heads keeps those batches on the memory-efficient
# kernel. CPU runs are left as they are because the CPU kernel handles enable_gqa natively.
_stock_use_gqa_in_sdpa = sdpa_attention.use_gqa_in_sdpa


def _use_gqa_in_sdpa(attention_mask, key, value):
    if key.is_cuda and not torch.backends.cuda.is_flash_attention_available():
        return False
    return _stock_use_gqa_in_sdpa(attention_mask, key, value)


sdpa_attention.use_gqa_in_sdpa = _use_gqa_in_sdpa
