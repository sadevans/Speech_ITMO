import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from melbanks import LogMelFilterBanks


def load_or_generate_audio(wav_path: str = None, duration: float = 2.0, sr: int = 16000):
    """
    Загружает аудиофайл или генерирует тестовый сигнал, если файл не указан.
    """
    if wav_path is not None:
        signal, sr = torchaudio.load(wav_path, format="wav")
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            signal = resampler(signal)
        sr = 16000
        print(f'Downloaded !')
    else:
        # generate test signal
        sr = 16000
        t = torch.linspace(0, duration, int(sr * duration))
        signal = (
            0.5 * torch.sin(2 * np.pi * 440 * t) + 
            0.3 * torch.sin(2 * np.pi * 880 * t) +
            0.2 * torch.randn_like(t) * 0.1
        )
        signal = signal.unsqueeze(0)
        print(f'Generated !')

    return signal, sr


def compute_spectrograms(signal, sr: int = 16000):
    """
    Вычисляет MelSpectrogram через torchaudio и через вашу реализацию.
    """
    native_melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80,
    )(signal)

    custom_logmel = LogMelFilterBanks()(signal)

    native_logmel = torch.log(native_melspec + 1e-6)

    return native_logmel, custom_logmel, native_melspec


def validate_implementations(native_logmel, custom_logmel, atol: float = 1e-5):
    """
    Проверяет совпадение реализаций и выводит метрики.
    """
    assert native_logmel.shape == custom_logmel.shape, \
        f'Shape mismatch: {native_logmel.shape} vs {custom_logmel.shape}'

    assert torch.allclose(native_logmel, custom_logmel, atol=atol), \
        'Values do not match! Check your implementation.'

    max_diff = torch.max(torch.abs(native_logmel - custom_logmel)).item()
    mean_diff = torch.mean(torch.abs(native_logmel - custom_logmel)).item()

    print(f'Shapes match: {native_logmel.shape}')
    print(f'Max absolute difference: {max_diff:.2e}')
    print(f'Mean absolute difference: {mean_diff:.2e}')

    return True


def plot_comparison(native_logmel, custom_logmel, native_melspec, sr: int = 16000):
    """
    Создаёт графики для отчета.
    """
    native_logmel_np = native_logmel.squeeze(0).detach().cpu().numpy()
    custom_logmel_np = custom_logmel.squeeze(0).detach().cpu().numpy()
    native_melspec_np = native_melspec.squeeze(0).detach().cpu().numpy()

    hop_length = 160
    time_axis = np.arange(native_logmel_np.shape[1]) * hop_length / sr

    mel_bins = np.arange(native_logmel_np.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparison: LogMelFilterBanks Implementation vs torchaudio', fontsize=16, fontweight='bold')

    im1 = axes[0, 0].imshow(
        10 * np.log10(native_melspec_np + 1e-6),
        aspect='auto', origin='lower', cmap='viridis',
        extent=[time_axis[0], time_axis[-1], mel_bins[0], mel_bins[-1]]
    )
    axes[0, 0].set_title('torchaudio.MelSpectrogram (dB)')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[0, 0], label='Amplitude [dB]')

    im2 = axes[0, 1].imshow(
        native_logmel_np,
        aspect='auto', origin='lower', cmap='viridis',
        extent=[time_axis[0], time_axis[-1], mel_bins[0], mel_bins[-1]]
    )
    axes[0, 1].set_title('LogMelFilterBanks Output (log)')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Mel Bin')
    plt.colorbar(im2, ax=axes[0, 1], label='log(Energy + 1e-6)')

    diff = np.abs(native_logmel_np - custom_logmel_np)
    im3 = axes[0, 2].imshow(
        diff,
        aspect='auto', origin='lower', cmap='hot',
        extent=[time_axis[0], time_axis[-1], mel_bins[0], mel_bins[-1]],
        vmin=0, vmax=1e-4
    )
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].set_xlabel('Time [s]')
    axes[0, 2].set_ylabel('Mel Bin')
    plt.colorbar(im3, ax=axes[0, 2], label='|Δ|')

    frame_idx = native_logmel_np.shape[1] // 2
    axes[1, 0].plot(mel_bins, native_logmel_np[:, frame_idx], label='torchaudio + log', linewidth=2)
    axes[1, 0].plot(mel_bins, custom_logmel_np[:, frame_idx], '--', label='LogMelFilterBanks', linewidth=2)
    axes[1, 0].set_title(f'Mel Spectrum at t={time_axis[frame_idx]:.2f}s')
    axes[1, 0].set_xlabel('Mel Bin')
    axes[1, 0].set_ylabel('log(Energy)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    mel_bin = 40
    axes[1, 1].plot(time_axis, native_logmel_np[mel_bin, :], label='torchaudio + log', linewidth=2)
    axes[1, 1].plot(time_axis, custom_logmel_np[mel_bin, :], '--', label='LogMelFilterBanks', linewidth=2)
    axes[1, 1].set_title(f'Energy Trajectory for Mel Bin #{mel_bin}')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('log(Energy)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].hist(diff.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Distribution of Absolute Differences')
    axes[1, 2].set_xlabel('|torchaudio - custom|')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./part1/figs/logmel_comparison_report.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    os.makedirs('./part1/figs/', exist_ok=True)
    WAV_PATH = './part1/data/test.wav'  # Укажите путь к файлу, например: 'audio/sample.wav'
    
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print('Loading/generating audio...')
    signal, sr = load_or_generate_audio(WAV_PATH)
    print(f'Signal shape: {signal.shape}, Sample rate: {sr} Hz')

    print('Computing spectrograms...')
    native_logmel, custom_logmel, native_melspec = compute_spectrograms(signal, sr)

    print('Validating implementations...')
    validate_implementations(native_logmel, custom_logmel)

    print('Generating comparison plots...')
    plot_comparison(native_logmel, custom_logmel, native_melspec, sr)
