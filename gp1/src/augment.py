from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

__all__ = [
    'WaveformAugment',
    'make_train_waveform_augment',
    'LogMelSpecAugment',
]


class WaveformAugment:
    def __init__(
        self,
        *,
        p_gain: float = 0.35,
        gain_db_range: float = 6.0,
        p_noise: float = 0.35,
        noise_std: float = 0.004,
    ) -> None:
        self.p_gain = p_gain
        self.gain_db_range = gain_db_range
        self.p_noise = p_noise
        self.noise_std = noise_std

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del sample_rate
        x = waveform
        if self.p_gain > 0 and torch.rand(1).item() < self.p_gain:
            db = (torch.rand(1).item() * 2.0 - 1.0) * self.gain_db_range
            gain = float(10.0 ** (db / 20.0))
            x = x * gain
        if self.p_noise > 0 and torch.rand(1).item() < self.p_noise:
            n = torch.randn_like(x) * self.noise_std
            x = x + n
        return x.clamp_(-1.0, 1.0)


def make_train_waveform_augment(
    *,
    p_gain: float = 0.35,
    gain_db_range: float = 6.0,
    p_noise: float = 0.35,
    noise_std: float = 0.004,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    aug = WaveformAugment(
        p_gain=p_gain,
        gain_db_range=gain_db_range,
        p_noise=p_noise,
        noise_std=noise_std,
    )
    return aug


class LogMelSpecAugment(nn.Module):
    def __init__(
        self,
        *,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_param: int = 20,
        time_mask_param: int = 50,
    ) -> None:
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if x.dim() != 3:
            raise ValueError(f'Expected [B, n_mels, T], got {tuple(x.shape)}')
        out = x.clone()
        B, F, T = out.shape
        device = out.device
        for b in range(B):
            fill = out[b].amin()
            for _ in range(self.num_freq_masks):
                if F <= 1:
                    break
                f_w = int(
                    torch.randint(
                        1, min(self.freq_mask_param, F) + 1, (1,), device=device
                    ).item()
                )
                f0 = int(torch.randint(0, F - f_w + 1, (1,), device=device).item())
                out[b, f0 : f0 + f_w, :] = fill
            for _ in range(self.num_time_masks):
                if T <= 1:
                    break
                t_w = int(
                    torch.randint(
                        1, min(self.time_mask_param, T) + 1, (1,), device=device
                    ).item()
                )
                t0 = int(torch.randint(0, T - t_w + 1, (1,), device=device).item())
                out[b, :, t0 : t0 + t_w] = fill
        return out
