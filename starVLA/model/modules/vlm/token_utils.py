from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def ensure_additional_special_tokens(
    model,
    tokenizer,
    tokens: Iterable[str],
    *,
    init_strategy: str = "normal",
) -> dict[str, int]:
    ordered_tokens: list[str] = []
    seen = set()
    for token in tokens:
        token = str(token).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered_tokens.append(token)

    if not ordered_tokens:
        return {}

    vocab = tokenizer.get_vocab()
    tokens_to_add = [token for token in ordered_tokens if token not in vocab]
    old_embed = model.get_input_embeddings()
    old_embed_size = old_embed.weight.shape[0]
    old_weight = old_embed.weight.data.detach().clone()

    added_now = 0
    if tokens_to_add:
        existing_additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        merged_additional = existing_additional + [token for token in tokens_to_add if token not in existing_additional]
        added_now = tokenizer.add_special_tokens({"additional_special_tokens": merged_additional})

    target_size = old_embed_size + added_now
    if target_size > old_embed_size:
        model.resize_token_embeddings(target_size)
        new_embed = model.get_input_embeddings()
        new_rows = new_embed.weight.data[old_embed_size:target_size]
        with torch.no_grad():
            if init_strategy == "avg":
                ref_vec = old_weight.mean(dim=0, keepdim=True)
                new_rows.copy_(ref_vec.expand_as(new_rows))
            elif init_strategy == "zero":
                new_rows.zero_()
            elif init_strategy == "normal":
                finite_old_weight = old_weight.float()[torch.isfinite(old_weight.float())]
                if finite_old_weight.numel() > 1:
                    std = float(finite_old_weight.std(unbiased=False).item())
                else:
                    std = float("nan")
                if not torch.isfinite(torch.tensor(std)):
                    std = 0.02
                nn.init.normal_(new_rows, mean=0.0, std=max(std, 1e-6))
            else:
                raise ValueError(f"Unknown init_strategy: {init_strategy}")

    mapping = {}
    for token in ordered_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if token_ids != [token_id]:
            raise RuntimeError(
                f"Token {token!r} is not encoded as a single dedicated token after registration: {token_ids}"
            )
        mapping[token] = int(token_id)
    return mapping
