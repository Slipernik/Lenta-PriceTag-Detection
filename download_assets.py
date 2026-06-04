from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path


FOLDER_URL = "https://drive.google.com/drive/folders/1TLLUYbAAXxDYL-VbxvluWZTu1D5_yItK?usp=sharing"
REPO_ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    REPO_ROOT / "weights" / "best.pt",
    *[REPO_ROOT / "test_img" / f"{idx}.png" for idx in range(1, 6)],
    *[REPO_ROOT / "test_video" / f"{idx}.mp4" for idx in range(1, 6)],
]


def find_missing_files() -> list[Path]:
    return [path for path in REQUIRED_FILES if not path.exists()]


def ensure_parent_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def collect_downloaded_files(root: Path) -> dict[str, Path]:
    files_by_name: dict[str, Path] = {}

    for path in root.rglob("*"):
        if path.is_file():
            files_by_name.setdefault(path.name, path)

    return files_by_name


def copy_missing_files(download_root: Path, missing_files: list[Path]) -> None:
    files_by_name = collect_downloaded_files(download_root)
    unresolved: list[Path] = []

    for target in missing_files:
        source = files_by_name.get(target.name)

        if source is None:
            unresolved.append(target)
            continue

        shutil.copy2(source, target)
        print(f"Downloaded {target.relative_to(REPO_ROOT)}")

    if unresolved:
        unresolved_text = "\n".join(f"- {path.relative_to(REPO_ROOT)}" for path in unresolved)
        raise FileNotFoundError(
            "Could not find all required files in the downloaded Google Drive folder:\n"
            f"{unresolved_text}"
        )


def download_folder_to_temp(tmpdir: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download missing assets. Install dependencies from requirements.txt first."
        ) from exc

    print("Missing files detected. Downloading assets from Google Drive...")
    result = gdown.download_folder(
        url=FOLDER_URL,
        output=str(tmpdir),
        quiet=False,
        remaining_ok=True,
    )

    if not result:
        raise RuntimeError("Google Drive download failed or returned no files.")


def main() -> int:
    missing_files = find_missing_files()

    if not missing_files:
        print("All required assets already exist. Nothing to download.")
        return 0

    ensure_parent_dirs(missing_files)

    with tempfile.TemporaryDirectory(prefix="detection_assets_") as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        download_folder_to_temp(tmpdir)
        copy_missing_files(tmpdir, missing_files)

    remaining_missing = find_missing_files()
    if remaining_missing:
        remaining_text = "\n".join(f"- {path.relative_to(REPO_ROOT)}" for path in remaining_missing)
        raise RuntimeError(f"Some required files are still missing after download:\n{remaining_text}")

    print("Asset download completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
