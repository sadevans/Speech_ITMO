from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .augment import make_train_waveform_augment
from .char_vocab import BLANK_IDX, encode_texts, greedy_ctc_decode
from .dataset import build_dataloaders
from .metrics import cer_by_speaker, cer_over_utterances, exact_match_rate, harmonic_mean
from .model import DigitCTCModel, count_parameters
from .text_normalize import TextNormalizationMode

__all__ = ['main', 'parse_args', 'run_training']


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train GP1 digit CTC baseline (TensorBoard, per-speaker CER).')
    root = Path(__file__).resolve().parents[2]
    data = root / 'data'
    p.add_argument('--train-csv', type=Path, default=data / 'train' / 'train.csv')
    p.add_argument('--train-root', type=Path, default=data / 'train')
    p.add_argument('--dev-csv', type=Path, default=data / 'dev' / 'dev.csv')
    p.add_argument('--dev-root', type=Path, default=data / 'dev')
    p.add_argument('--log-dir', type=Path, default=root / 'runs' / 'ctc_baseline')
    p.add_argument('--checkpoint-dir', type=Path, default=root / 'checkpoints')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--text-mode',
        type=str,
        choices=('digits', 'words'),
        default='digits',
        help='CTC target: digit string (recommended for this baseline) or Russian words.',
    )
    p.add_argument('--no-augment', action='store_true', help='Disable waveform gain/noise augment.')
    p.add_argument('--no-spec-augment', action='store_true', help='Disable SpecAugment in the model.')
    p.add_argument('--no-tensorboard', action='store_true')
    p.add_argument('--no-cudnn-benchmark', action='store_true')
    p.add_argument('--resume', type=Path, default=None, help='Path to checkpoint .pt to resume weights.')
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_speakers(train_csv: Path) -> set[str]:
    df = pd.read_csv(train_csv)
    return set(df['spk_id'].astype(str).unique())


def greedy_decode_batch(log_probs: torch.Tensor) -> list[str]:
    t_steps, batch_size, _ = log_probs.shape
    out: list[str] = []
    for b in range(batch_size):
        idx = log_probs[:t_steps, b, :].argmax(dim=-1).detach().cpu().tolist()
        out.append(greedy_ctc_decode(idx))
    return out


def _split_refs_hyps_by_spk_in_train(
    refs: list[str],
    hyps: list[str],
    spk_ids: list[str],
    train_speakers: set[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    in_r, in_h, out_r, out_h = [], [], [], []
    for r, h, s in zip(refs, hyps, spk_ids):
        if s in train_speakers:
            in_r.append(r)
            in_h.append(h)
        else:
            out_r.append(r)
            out_h.append(h)
    return in_r, in_h, out_r, out_h


def evaluate_batch_metrics(
    refs: list[str],
    hyps: list[str],
    spk_ids: list[str],
    genders: list[str],
    train_speakers: set[str],
) -> dict[str, float]:
    cer_mean = cer_over_utterances(refs, hyps)
    acc = exact_match_rate(refs, hyps)
    in_r, in_h, ood_r, ood_h = _split_refs_hyps_by_spk_in_train(
        refs, hyps, spk_ids, train_speakers
    )
    cer_ind = cer_over_utterances(in_r, in_h) if in_r else 0.0
    cer_ood = cer_over_utterances(ood_r, ood_h) if ood_r else 0.0
    if in_r and ood_r:
        hmean = harmonic_mean(cer_ind, cer_ood)
    elif in_r:
        hmean = cer_ind
    elif ood_r:
        hmean = cer_ood
    else:
        hmean = 0.0

    out: dict[str, float] = {
        'cer/mean': cer_mean,
        'cer/hmean_in_ood': hmean,
        'cer/in_domain': cer_ind,
        'cer/out_of_domain': cer_ood,
        'seq_accuracy': acc,
    }
    per_spk = cer_by_speaker(refs, hyps, spk_ids)
    for spk, v in sorted(per_spk.items()):
        out[f'cer/spk/{spk}'] = v
    per_g = cer_by_speaker(refs, hyps, genders)
    for g, v in sorted(per_g.items()):
        out[f'cer/gender/{g}'] = v
    return out


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc: nn.CTCLoss,
    device: torch.device,
    grad_clip: float,
    writer: SummaryWriter | None,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        waves = batch['waveform'].to(device)
        wlen = batch['waveform_length'].to(device)
        texts = batch['text']
        targets, target_lengths = encode_texts(texts)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        log_probs, out_lens = model(waves, wlen)
        loss = ctc(log_probs, targets, out_lens, target_lengths)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_v = float(loss.detach().item())
        total_loss += loss_v
        n_batches += 1
        if writer is not None:
            writer.add_scalar('train/loss_step', loss_v, global_step)
        global_step += 1
    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    ctc: nn.CTCLoss,
    device: torch.device,
    train_speakers: set[str],
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_refs: list[str] = []
    all_hyps: list[str] = []
    all_spk: list[str] = []
    all_gen: list[str] = []

    for batch in loader:
        waves = batch['waveform'].to(device)
        wlen = batch['waveform_length'].to(device)
        texts = batch['text']
        targets, target_lengths = encode_texts(texts)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        log_probs, out_lens = model(waves, wlen)
        loss = ctc(log_probs, targets, out_lens, target_lengths)
        total_loss += float(loss.item())
        n_batches += 1

        hyps = greedy_decode_batch(log_probs)
        all_hyps.extend(hyps)
        all_refs.extend(batch['reference_digits'])
        all_spk.extend(str(s) for s in batch['spk_id'])
        all_gen.extend(str(g) for g in batch['gender'])

    mean_loss = total_loss / max(n_batches, 1)
    metrics = evaluate_batch_metrics(all_refs, all_hyps, all_spk, all_gen, train_speakers)
    metrics['loss'] = mean_loss
    return mean_loss, metrics


def run_training(args: argparse.Namespace) -> None:
    if args.text_mode != 'digits':
        raise ValueError(
            'This baseline uses a digit-only CTC head (char_vocab). '
            'Pass --text-mode digits or extend the vocabulary and encode_texts().'
        )
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Указан --device cuda, но torch.cuda.is_available() == False. '
            )
        if not args.no_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(device)
        print(f'[train] Устройство: CUDA {device} — {name}')
    else:
        print(
            '[train] Устройство: CPU — обучение будет медленным; '
            'для GPU нужен PyTorch с CUDA и доступная видеокарта.'
        )

    if device.type == 'cuda' and args.num_workers == 0:
        print(
            '[train] Подсказка: декодирование wav/mp3 и ресемплинг идут в DataLoader на CPU. '
            'При высокой загрузке CPU попробуйте --num-workers 4 (или 8).'
        )

    train_speakers = load_train_speakers(args.train_csv)

    train_tf = None if args.no_augment else make_train_waveform_augment()
    text_mode: TextNormalizationMode = 'digits'
    loader_train, loader_dev = build_dataloaders(
        args.train_csv,
        args.train_root,
        args.dev_csv,
        args.dev_root,
        text_mode=text_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_tf,
        dev_transform=None,
    )

    model = DigitCTCModel(use_spec_augment=not args.no_spec_augment).to(device)
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model_state_dict'])

    n_params = count_parameters(model)
    ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(args.log_dir))
        writer.add_scalar('meta/num_parameters', float(n_params), 0)

    with open(args.log_dir / 'hparams.txt', 'w', encoding='utf-8') as f:
        f.write(repr(vars(args)))

    metrics_path = args.log_dir / 'metrics.jsonl'
    global_step = 0
    best_cer = float('inf')

    for epoch in range(1, args.epochs + 1):
        tr_loss, global_step = train_one_epoch(
            model,
            loader_train,
            optimizer,
            ctc,
            device,
            args.grad_clip,
            writer,
            global_step,
        )
        scheduler.step()
        dev_loss, dev_metrics = validate(model, loader_dev, ctc, device, train_speakers)

        if writer is not None:
            writer.add_scalar('train/loss_epoch', tr_loss, epoch)
            writer.add_scalar('optim/lr', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('val/loss', dev_loss, epoch)
            for k, v in dev_metrics.items():
                if k == 'loss':
                    continue
                writer.add_scalar(f'val/{k}', v, epoch)

        row = {'epoch': epoch, 'train_loss': tr_loss, **{f'val_{k}': v for k, v in dev_metrics.items()}}
        with open(metrics_path, 'a', encoding='utf-8') as jf:
            jf.write(json.dumps(row, ensure_ascii=False) + '\n')

        cer_key = dev_metrics.get('cer/mean', 0.0)
        if cer_key < best_cer:
            best_cer = cer_key
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer_mean': best_cer,
                'args': vars(args),
            }
            torch.save(ckpt, args.checkpoint_dir / 'best.pt')

        print(
            f'epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  '
            f'val_loss={dev_loss:.4f}  val_cer_mean={dev_metrics["cer/mean"]:.4f}  '
            f'val_cer_hmean_in_ood={dev_metrics["cer/hmean_in_ood"]:.4f}  '
            f'val_acc={dev_metrics["seq_accuracy"]:.4f}'
        )

    if writer is not None:
        writer.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == '__main__':
    main()
