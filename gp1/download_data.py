from __future__ import annotations
import argparse
import shutil
import zipfile
from pathlib import Path
import os

import gdown

TRAIN_FILE_ID = '15CpIWvVDA6mOlPxyI4-vicyXSqd-EcIb'
DEV_FILE_ID = '1Jlw09RSJjhJTxdN3VQj5Bph4zRNwOqSL'


def _extract_if_archive(path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffixes = path.suffixes
    if suffixes[-2:] == ['.tar', '.gz'] or path.suffix == '.tgz':
        shutil.unpack_archive(path, extract_dir=dest_dir)
    elif path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(dest_dir)
    else:
        raise RuntimeError(
            f'Неизвестный формат архива: {path.name}. Распакуйте вручную.'
        )


def download_one(file_id: str, out_dir: Path, fuzzy: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    # Каталог + разделитель → gdown подставляет исходное имя файла с Drive
    target_dir = str(out_dir.resolve()) + os.sep
    saved = gdown.download(url, target_dir, quiet=False, fuzzy=fuzzy)
    if not saved:
        raise RuntimeError(f'Не удалось скачать {file_id}')
    return Path(saved)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--out-dir',
        type=Path,
        default=Path('data/gp1_raw'),
        help='Куда сохранить .zip/.tar.gz',
    )
    p.add_argument(
        '--extract-dir',
        type=Path,
        default=Path('data/gp1'),
        help='Куда распаковать (общая папка для обоих архивов)',
    )
    p.add_argument(
        '--no-extract',
        action='store_true',
        help='Только скачать, не распаковывать',
    )
    p.add_argument(
        '--fuzzy',
        action='store_true',
        help='Использовать fuzzy-режим gdown (если обычная загрузка падает)',
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    archives = [
        ('train', TRAIN_FILE_ID),
        ('dev', DEV_FILE_ID),
    ]

    for name, fid in archives:
        print(f'Downloading {name} …')
        archive_path = download_one(fid, args.out_dir, fuzzy=args.fuzzy)

        if not args.no_extract:
            sub = args.extract_dir / name
            print(f'Extracting {archive_path.name} → {sub} …')
            _extract_if_archive(archive_path, sub)

    print('Done.')


if __name__ == '__main__':
    main()