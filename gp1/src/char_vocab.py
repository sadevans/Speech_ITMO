from __future__ import annotations

import torch

BLANK_IDX = 10
VOCAB = '0123456789'
NUM_CLASSES = len(VOCAB) + 1  # + blank

_char_to_idx = {c: i for i, c in enumerate(VOCAB)}
_idx_to_char = {i: c for i, c in enumerate(VOCAB)}


def encode_texts(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    flat: list[int] = []
    lengths: list[int] = []
    for t in texts:
        for ch in t:
            if ch not in _char_to_idx:
                raise ValueError(f'Invalid char {ch!r} in label {t!r}')
            flat.append(_char_to_idx[ch])
        lengths.append(len(t))
    if not flat:
        raise ValueError('Empty batch texts')
    targets = torch.tensor(flat, dtype=torch.long)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return targets, target_lengths


def greedy_ctc_decode(indices: list[int], blank: int = BLANK_IDX) -> str:
    out: list[str] = []
    prev: int | None = None
    for idx in indices:
        if idx == blank:
            prev = None
            continue
        if idx == prev:
            continue
        prev = idx
        if 0 <= idx < len(VOCAB):
            out.append(_idx_to_char[idx])
    return ''.join(out)
