"""Download the MovieLens ml-latest-small dataset.

Usage:
    python download_data.py            # fetch + verify (skips if already present)
    python download_data.py --force    # re-fetch even if present
    python download_data.py --mirror   # skip the official host, go straight to mirror

Writes to api/data/raw/ml-latest-small/, which is gitignored -- the dataset is
reproducible from this script, so it doesn't belong in git.

TWO SOURCES, IN ORDER:
  1. The official GroupLens zip.
  2. A per-file mirror, used only if (1) fails.

As of 2026-09-11 the official host files.grouplens.org has an EXPIRED TLS
certificate (it lapsed 2026-08-28), so (1) fails and this script falls back to
(2) automatically. Try (1) again later -- it'll start working the moment
GroupLens renews, and this script needs no change.

Every mirrored file is checked against a SHA-256 pinned below, and the row
counts are checked against the dataset's published statistics. That is why this
script never disables TLS verification: pinned hashes verify the *bytes*, which
is strictly stronger than trusting the transport. If you find advice online
telling you to pass `verify=False` or `ssl._create_unverified_context()` to get
around the expired cert -- don't. Verify the content instead.

Dataset: https://grouplens.org/datasets/movielens/
Cite as: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens
Datasets: History and Context. ACM TiiS 5, 4: 19:1-19:19.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

OFFICIAL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Mirror of the individual CSVs. Verified on 2026-09-11 to be byte-identical
# across two independent copies of the dataset, and to match the published
# row counts below. The hashes are what make this safe to use.
MIRROR_BASE = "https://raw.githubusercontent.com/smanihwr/ml-latest-small/master"

SHA256 = {
    "ratings.csv": "80da8b3393dae325bbba5a31f291a6ba55d8d4f4396de3c456f2c1635b1b70e8",
    "movies.csv": "c47b783f23eb304413307e1bb05335e427737da20ec8d3c0e96335567dd5f3dc",
    "links.csv": "028a4a1d17751ca9276e0e0718cfc22356350f224a996e1ec348c23274b9209d",
}

# Published statistics for ml-latest-small. If these ever fail, we did not get
# the dataset we think we got -- fail loudly here rather than produce a
# confusing KeyError three scripts later.
EXPECTED_COLUMNS = {
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "movies.csv": ["movieId", "title", "genres"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
}
EXPECTED_ROWS = {"ratings.csv": 100_836, "movies.csv": 9_742, "links.csv": 9_742}

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = RAW_DIR / "ml-latest-small"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    pct = min(100.0, block_num * block_size * 100 / total_size)
    print(f"\r    {pct:5.1f}%", end="", flush=True)


def try_official() -> bool:
    """Attempt the official zip. Returns True on success, False to fall back."""
    print(f"[1/2] official host: {OFFICIAL_URL}")
    try:
        urllib.request.urlretrieve(OFFICIAL_URL, ZIP_PATH, reporthook=show_progress)
    except (urllib.error.URLError, OSError) as err:
        ZIP_PATH.unlink(missing_ok=True)  # don't leave a partial file behind
        reason = getattr(err, "reason", err)
        print(f"\n    unavailable: {reason}")
        if "CERTIFICATE_VERIFY_FAILED" in str(reason):
            print("    (expected -- GroupLens' certificate expired 2026-08-28)")
        return False

    print(f"\n    got {ZIP_PATH.stat().st_size / 1e6:.1f} MB, extracting")
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
    except zipfile.BadZipFile:
        ZIP_PATH.unlink(missing_ok=True)
        print("    zip was corrupt, falling back to mirror")
        return False
    return True


def fetch_mirror() -> None:
    """Fetch the CSVs individually, verifying each against its pinned hash."""
    print(f"[2/2] mirror: {MIRROR_BASE}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for name, expected_hash in SHA256.items():
        dest = DATASET_DIR / name
        url = f"{MIRROR_BASE}/{name}"
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.URLError, OSError) as err:
            dest.unlink(missing_ok=True)
            sys.exit(f"    {name}: download failed: {getattr(err, 'reason', err)}")

        actual = sha256_of(dest)
        if actual != expected_hash:
            dest.unlink()  # never leave unverified data on disk
            sys.exit(
                f"    {name}: SHA-256 MISMATCH -- file deleted.\n"
                f"      expected {expected_hash}\n"
                f"      actual   {actual}\n"
                f"    Do not use this file. The mirror may have changed; "
                f"re-check against the official dataset."
            )
        size_kb = dest.stat().st_size / 1024
        print(f"    {name:<12} {size_kb:>8.0f} KB  sha256 ok")


def verify() -> None:
    """Check structure and row counts against the published statistics."""
    print("verifying:")
    problems = []

    for name, columns in EXPECTED_COLUMNS.items():
        path = DATASET_DIR / name
        if not path.exists():
            problems.append(f"missing file: {name}")
            continue

        with path.open() as f:
            header = f.readline().strip().split(",")
            n_rows = sum(1 for _ in f)

        if missing := [c for c in columns if c not in header]:
            problems.append(f"{name}: missing columns {missing} (found {header})")
            continue
        if n_rows != EXPECTED_ROWS[name]:
            problems.append(
                f"{name}: {n_rows:,} rows, expected {EXPECTED_ROWS[name]:,}"
            )
            continue
        print(f"    {name:<12} {n_rows:>7,} rows  {columns}")

    if problems:
        sys.exit("\nverification FAILED:\n    " + "\n    ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch even if present")
    parser.add_argument(
        "--mirror", action="store_true", help="skip the official host entirely"
    )
    args = parser.parse_args()

    if DATASET_DIR.exists() and not args.force:
        print(f"already present, skipping fetch: {DATASET_DIR}")
        print("(use --force to re-download)")
        verify()
        return

    if args.force and DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not (not args.mirror and try_official()):
        fetch_mirror()

    verify()
    print(f"\nready: {DATASET_DIR}")
    print("next:  jupyter lab notebooks/01_explore.ipynb")


if __name__ == "__main__":
    main()
