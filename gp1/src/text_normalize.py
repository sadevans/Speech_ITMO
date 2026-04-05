from __future__ import annotations

import re
from typing import Literal

from num2words import num2words

TextNormalizationMode = Literal['digits', 'words']


def parse_transcription_number(raw: str) -> int:
    s = str(raw).strip().replace('_', '').replace(' ', '')
    if not s.isdigit():
        raise ValueError(f'Expected digit transcription, got {raw!r}')
    return int(s)


def standardize_russian_words(phrase: str) -> str:
    s = phrase.strip().lower()
    return re.sub(r'\s+', ' ', s)


def normalize_transcription(
    raw: str,
    mode: TextNormalizationMode,
    *,
    min_value: int = 0,
    max_value: int = 99_999_999,
) -> tuple[str, str]:
    n = parse_transcription_number(raw)
    if not (min_value <= n <= max_value):
        raise ValueError(
            f'Transcription {n} out of allowed range [{min_value}, {max_value}]'
        )
    reference_digits = str(n)
    if mode == 'digits':
        return reference_digits, reference_digits
    if mode == 'words':
        spoken = num2words(n, lang='ru', to='cardinal')
        return standardize_russian_words(spoken), reference_digits
    raise ValueError(f'Unknown text normalization mode: {mode!r}')


def denormalize_digits_for_submission(
    digit_hypothesis: str,
    *,
    empty_fallback: int = 1000,
    min_value: int = 0,
    max_value: int = 99_999_999,
) -> int:
    s = ''.join(ch for ch in str(digit_hypothesis) if ch.isdigit())
    if not s:
        return int(empty_fallback)
    n = int(s)
    if n < min_value:
        return int(min_value)
    if n > max_value:
        return int(max_value)
    return int(n)
