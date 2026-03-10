from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,                  # size of FFT, creates n_fft // 2 + 1 bins
            samplerate: int = 16000,           # sample rate of audio signal
            hop_length: int = 160,             # length of hop between STFT windows
            n_mels: int = 80,                  # number of mel filterbanks
            pad_mode: str = 'reflect',         # padding method 
            power: float = 2.0,                # exponent for the magnitude spectogram
            normalize_stft: bool = False,      # normalize by magnitude after stft
            onesided: bool = True,
            center: bool = True,               # whether to pad
            return_complex: bool = True,
            f_min_hz: float = 0.0,             # min frequency
            f_max_hz: Optional[float] = None,  # max frequency
            norm_mel: Optional[str] = None,    # if "slaney", divide the triangular mel weights by the width of the mel band
            mel_scale: str = 'htk'             # scale to use
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        # Do correct initialization of stft params below:
        # hop_length, n_mels, center, return_complex, onesided, normalize_stft, pad_mode, power
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        # Do correct initialization of mel fbanks params below:
        # f_min_hz, f_max_hz, norm_mel, mel_scale
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz if self.f_max_hz is not None else float(self.samplerate // 2),
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale,
        )

    def spectrogram(self, x):
        # x - is an input signal
        return torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        # Return log mel filterbanks matrix
        spec = self.spectrogram(x)
        power_spec = spec.abs().pow(self.power)

        mel_spec = torch.matmul(
            power_spec.transpose(1, 2),  # (batch, n_frames, n_freqs)
            self.mel_fbanks              # (n_freqs, n_mels)
        ).transpose(1, 2)

        log_mel = torch.log(mel_spec + 1e-6)

        return log_mel