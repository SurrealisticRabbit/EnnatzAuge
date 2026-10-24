"""Download SoccerNet SN-GSR-2025.

Only the split archives are fetched, and only those that are still needed: the
source repository is ~35 GB of zips, which is more than many hosts want to hold
at once. ``download_split`` pulls one archive at a time so a caller can convert
and delete it before fetching the next.
"""

from __future__ import annotations

from pathlib import Path

REPO_ID = "SoccerNet/SN-GSR-2025"
SPLITS = ("train", "valid", "test", "challenge")
LABELLED_SPLITS = ("train", "valid", "test")  # challenge ships no ground truth


def download_split(split: str, raw_dir: Path) -> Path:
    """Fetch ``<split>.zip`` into ``raw_dir`` and return its path."""
    from huggingface_hub import hf_hub_download

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{split}.zip"
    if target.exists():
        print(f"  {split}.zip already present "
              f"({target.stat().st_size / 1e9:.1f} GB)", flush=True)
        return target

    print(f"  downloading {split}.zip from {REPO_ID} ...", flush=True)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="main",
        filename=f"{split}.zip",
        local_dir=str(raw_dir),
    )
    # hf_hub_download can return a cache path when local_dir is not honoured.
    downloaded = Path(downloaded)
    if downloaded.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        downloaded.replace(target)
    return target


def extract_split(zip_path: Path, data_root: Path, split: str) -> Path:
    """Unpack a split archive into ``data_root/<split>``.

    The full extraction is ~35 GB across all splits. :mod:`convert` reads
    straight from the archive instead when disk is tight.
    """
    import zipfile

    out_dir = data_root / split
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  {split}: already extracted at {out_dir}", flush=True)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path.name} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    return out_dir
