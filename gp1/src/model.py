from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .augment import LogMelSpecAugment
from .char_vocab import NUM_CLASSES

__all__ = [
    'DigitCTCModel',
    'count_parameters',
    'cnn_time_length',
    'mel_frames_from_waveform_samples',
]


def mel_frames_from_waveform_samples(
    n_samples: int,
    *,
    n_fft: int,
    hop_length: int,
    center: bool,
) -> int:
    n = int(n_samples)
    if center:
        n = n + 2 * (n_fft // 2)
    return max(1, (n - n_fft) // hop_length + 1)


def _conv1d_out_len(
    length: int,
    *,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    if length < 1:
        return 0
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


_CNN_LAYERS = (
    (64, 3, 1, 1),
    (128, 3, 2, 1),
    (256, 3, 2, 1),
    (256, 3, 2, 1),
)


def cnn_time_length(mel_time_steps: int) -> int:
    t = mel_time_steps
    for _out_ch, k, s, p in _CNN_LAYERS:
        t = _conv1d_out_len(t, kernel=k, stride=s, padding=p)
    return max(t, 1)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class DigitCTCModel(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        dropout: float = 0.15,
        use_spec_augment: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = True

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=self.center,
            normalized=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        self.spec_augment = LogMelSpecAugment() if use_spec_augment else None

        layers: list[nn.Module] = []
        prev_ch = n_mels
        for out_ch, k, s, p in _CNN_LAYERS:
            layers.extend(
                [
                    nn.Conv1d(prev_ch, out_ch, k, stride=s, padding=p, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ]
            )
            prev_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        cnn_out = _CNN_LAYERS[-1][0]
        self.gru = nn.GRU(
            cnn_out,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * gru_hidden, NUM_CLASSES)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(p)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    def mel_frames_from_waveform_length(self, n_samples: torch.Tensor | int) -> torch.Tensor:
        if isinstance(n_samples, int):
            return torch.tensor(
                mel_frames_from_waveform_samples(
                    n_samples,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    center=self.center,
                ),
                dtype=torch.long,
            )
        ns = n_samples.long()
        if self.center:
            ns = ns + 2 * (self.n_fft // 2)
        return torch.clamp((ns - self.n_fft) // self.hop_length + 1, min=1)

    def cnn_lengths_from_mel(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        out = mel_lengths.long().clone()
        for _, k, s, p in _CNN_LAYERS:
            out = torch.clamp(
                (out + 2 * p - (k - 1) - 1) // s + 1,
                min=0,
            )
        return torch.clamp(out, min=1)

    def forward_log_probs(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() != 3 or waveform.size(1) != 1:
            raise ValueError(f'Expected waveform [B, 1, T], got {tuple(waveform.shape)}')

        x = self.mel(waveform)
        # torchaudio returns [B, 1, n_mels, T] for input [B, 1, T]; CNN expects [B, n_mels, T].
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        x = self.amplitude_to_db(x)
        if self.spec_augment is not None:
            x = self.spec_augment(x)

        x = self.cnn(x)
        time_after_cnn = x.size(-1)
        # [B, C, T] -> [T, B, C]
        x = x.permute(2, 0, 1)
        mel_lens = self.mel_frames_from_waveform_length(waveform_length).to(x.device)
        cnn_lens = self.cnn_lengths_from_mel(mel_lens)
        cnn_lens = torch.minimum(
            cnn_lens,
            torch.full_like(cnn_lens, time_after_cnn),
        )
        cnn_lens = torch.clamp(cnn_lens, min=1)

        x = self.dropout(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            cnn_lens.cpu(),
            enforce_sorted=False,
        )
        y, _ = self.gru(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y)
        y = self.dropout(y)
        logits = self.fc(y)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, cnn_lens

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_log_probs(waveform, waveform_length)
