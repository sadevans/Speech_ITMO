from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .text_normalize import TextNormalizationMode, normalize_transcription

__all__ = [
    'SpokenNumbersDataset',
    'collate_spoken_numbers',
    'build_dataloaders',
]


class SpokenNumbersDataset(Dataset):
    _required_columns = (
        'filename',
        'transcription',
        'spk_id',
        'gender',
        'ext',
        'samplerate',
    )

    def __init__(
        self,
        csv_path: str | Path,
        audio_root: str | Path,
        *,
        text_mode: TextNormalizationMode = 'digits',
        target_sample_rate: int = 16_000,
        transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.audio_root = Path(audio_root)
        self.text_mode: TextNormalizationMode = text_mode
        self.target_sample_rate = int(target_sample_rate)
        self.transform = transform

        self._df = pd.read_csv(self.csv_path)
        missing = [c for c in self._required_columns if c not in self._df.columns]
        if missing:
            raise ValueError(
                f'CSV {self.csv_path} missing columns {missing}; '
                f'expected {self._required_columns}'
            )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx]
        path = self.audio_root / str(row['filename'])
        if not path.is_file():
            raise FileNotFoundError(f'Audio not found: {path}')

        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        text, reference_digits = normalize_transcription(
            str(row['transcription']), self.text_mode
        )

        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.target_sample_rate
            )

        if self.transform is not None:
            waveform = self.transform(waveform, self.target_sample_rate)

        return {
            'waveform': waveform,
            'sample_rate': self.target_sample_rate,
            'text': text,
            'reference_digits': reference_digits,
            'spk_id': str(row['spk_id']),
            'gender': str(row['gender']),
            'ext': str(row['ext']),
            'manifest_samplerate': int(row['samplerate']),
            'path': str(path),
        }


def collate_spoken_numbers(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    waves = [b['waveform'] for b in batch]
    lengths = torch.tensor([w.shape[-1] for w in waves], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = [F.pad(w, (0, max_len - w.shape[-1])) for w in waves]
    return {
        'waveform': torch.stack(padded, dim=0),
        'waveform_length': lengths,
        'text': [b['text'] for b in batch],
        'reference_digits': [b['reference_digits'] for b in batch],
        'spk_id': [b['spk_id'] for b in batch],
        'gender': [b['gender'] for b in batch],
        'path': [b['path'] for b in batch],
    }


def build_dataloaders(
    train_csv: str | Path,
    train_audio_root: str | Path,
    dev_csv: str | Path,
    dev_audio_root: str | Path,
    *,
    text_mode: TextNormalizationMode = 'digits',
    target_sample_rate: int = 16_000,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    dev_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> tuple[DataLoader, DataLoader]:
    ds_train = SpokenNumbersDataset(
        train_csv,
        train_audio_root,
        text_mode=text_mode,
        target_sample_rate=target_sample_rate,
        transform=train_transform,
    )
    ds_dev = SpokenNumbersDataset(
        dev_csv,
        dev_audio_root,
        text_mode=text_mode,
        target_sample_rate=target_sample_rate,
        transform=dev_transform,
    )
    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_spoken_numbers,
        drop_last=False,
    )
    loader_dev = DataLoader(
        ds_dev,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_spoken_numbers,
        drop_last=False,
    )
    return loader_train, loader_dev
