from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

from fraud_aml.config import get_settings
from fraud_aml.logging_config import configure_logging, get_logger

COMPETITION = "ieee-fraud-detection"
FILES = ("train_transaction.csv", "train_identity.csv")


def _api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _human(size: float) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def manifest(api: Any | None = None) -> list[tuple[str, int]]:
    api = api or _api()
    listing = api.competition_list_files(COMPETITION)
    items = getattr(listing, "files", listing)
    sizes: dict[str, int] = {}
    for item in items:
        size = int(getattr(item, "totalBytes", 0) or getattr(item, "size", 0) or 0)
        sizes[str(item.name)] = size
    return [(name, sizes.get(name, -1)) for name in FILES]


def _extract_if_zipped(raw_dir: Path, name: str) -> None:
    zipped = raw_dir / f"{name}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as archive:
            archive.extractall(raw_dir)
        zipped.unlink()


def download(raw_dir: Path, *, yes: bool, api: Any | None = None) -> None:
    api = api or _api()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (raw_dir / name).exists():
            continue
        if not yes:
            continue
        api.competition_download_file(COMPETITION, name, path=str(raw_dir))
        _extract_if_zipped(raw_dir, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download IEEE-CIS fraud data (train files only).")
    parser.add_argument(
        "--yes", action="store_true", help="actually download (otherwise manifest only)"
    )
    parser.add_argument(
        "--raw-dir", default=None, help="where to put CSV files (default <data_dir>/raw)"
    )
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    log = get_logger("ingest")
    settings = get_settings()
    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.data_dir / "raw"

    api = _api()
    total = 0
    print(f"IEEE-CIS ({COMPETITION}) - files to download (to {raw_dir}):")
    for name, size in manifest(api):
        total += max(size, 0)
        print(f"  {name:28s} {_human(size):>10s}")
    print(f"  {'TOTAL':28s} {_human(total):>10s}\n")

    if not args.yes:
        print("Dry-run. Add --yes to download.")
        return 0

    download(raw_dir, yes=True, api=api)
    log.info("ingest_done", raw_dir=str(raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
