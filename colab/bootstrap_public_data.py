"""Prepare public Sleipner data in a fresh Colab workspace.

This script downloads required public Sleipner seismic vintages and benchmark
reference archives from CO2DataShare, extracts the needed files, and writes the
paths expected by the existing configs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


SEISMIC_RESOURCES = {
    "94p07": {
        "url": "https://co2datashare.org/dataset/sleipner-4d-seismic-dataset/resource/cd11f5e3-6162-42a7-b44d-6a126c7097da/download/94p07.zip",
        "zip_name": "94p07.zip",
        "target_name": "94p07mid.sgy",
        "member_hint": "94p07mid",
    },
    "01p07": {
        "url": "https://co2datashare.org/dataset/sleipner-4d-seismic-dataset/resource/2f874e5b-d06f-4e71-b334-19f97aec8fc5/download/01p07.zip",
        "zip_name": "01p07.zip",
        "target_name": "01p07mid.sgy",
        "member_hint": "01p07mid",
    },
    "04p07": {
        "url": "https://co2datashare.org/dataset/sleipner-4d-seismic-dataset/resource/cf417e18-17cd-4f89-9dd6-d641836a096b/download/04p07.zip",
        "zip_name": "04p07.zip",
        "target_name": "04p07mid.sgy",
        "member_hint": "04p07mid",
    },
    "06p07": {
        "url": "https://co2datashare.org/dataset/sleipner-4d-seismic-dataset/resource/2e2b6532-b4d6-48b0-aaa6-e9843e2b3862/download/06p07.zip",
        "zip_name": "06p07.zip",
        "target_name": "06p07mid.sgy",
        "member_hint": "06p07mid",
    },
    "94p10": {
        "url": "https://co2datashare.org/dataset/sleipner-4d-seismic-dataset/resource/8c86fba9-8a26-4933-b79a-4ac565063045/download/94p10.zip",
        "zip_name": "94p10.zip",
        "target_name": "94p10mid.sgy",
        "member_hint": "94p10mid",
    },
    "10p10": {
        "url": "https://co2datashare.org/dataset/sleipner-4d-seismic-dataset/resource/28d666ec-8b0a-47b6-a37b-645d9f07b0d9/download/10p10.zip",
        "zip_name": "10p10.zip",
        "target_name": "10p10mid.sgy",
        "member_hint": "10p10mid",
    },
}

BENCHMARK_ARCHIVES = {
    "velocities": {
        "url": "https://co2datashare.org/dataset/e6f67cbd-abf3-4d85-a118-ed386a994c2c/resource/5506f2f5-17e0-4d73-857a-6dba32fd415c/download/velocities_trends_surfaces.zip",
        "zip_name": "velocities_trends_surfaces.zip",
        "expected_link": Path("/tmp/sleipner_benchmark_restore/velocities_trends_surfaces/data"),
        "required_subdirs": ["DepthSurfaces_Grid", "HUM_Interval_Velocity_Trends"],
    },
    "plumes": {
        "url": "https://co2datashare.org/dataset/e6f67cbd-abf3-4d85-a118-ed386a994c2c/resource/feeaa90a-a32b-40f1-9990-c6acb2f6ce85/download/sleipner_plumes_boundaries.zip",
        "zip_name": "sleipner_plumes_boundaries.zip",
        "expected_link": Path("/tmp/sleipner_benchmark_restore/Sleipner_Plumes_Boundaries/data"),
        "required_subdirs": [f"L{i}" for i in range(1, 10)],
    },
}


def download_file(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        print(f"[skip] {destination.name} already exists")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {destination.name}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _extract_member_to_path(archive_path: Path, member_name: str, output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"[skip] {output_path.name} already exists")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    print(f"[write] {output_path}")


def prepare_seismic_exports(workspace: Path, *, force: bool) -> None:
    raw_dir = workspace / "examples" / "exports" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, resource in SEISMIC_RESOURCES.items():
        zip_path = raw_dir / resource["zip_name"]
        download_file(resource["url"], zip_path, force=force)
        target_path = raw_dir / resource["target_name"]
        with zipfile.ZipFile(zip_path) as archive:
            candidate_members = [
                member
                for member in archive.namelist()
                if resource["member_hint"].lower() in Path(member).name.lower()
            ]
            if not candidate_members:
                raise FileNotFoundError(
                    f"Could not locate member containing '{resource['member_hint']}' inside {zip_path}."
                )
            candidate_members.sort(key=lambda member: len(Path(member).name))
            member = candidate_members[0]
        _extract_member_to_path(zip_path, member, target_path, force=force)
        print(f"[ok] {name} -> {target_path.name}")


def _find_tree_root_with_subdirs(extract_dir: Path, required_subdirs: list[str]) -> Path:
    required = set(required_subdirs)
    for candidate in extract_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        children = {child.name for child in candidate.iterdir() if child.is_dir()}
        if required.issubset(children):
            return candidate
    raise FileNotFoundError(
        f"Could not find extracted directory containing all required subdirectories: {sorted(required)}"
    )


def _symlink_or_copy(source: Path, destination: Path, *, force: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not force:
            print(f"[skip] {destination} already exists")
            return
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)
    print(f"[link] {destination} -> {source}")


def prepare_benchmark_archives(*, force: bool) -> None:
    root = Path("/tmp/sleipner_benchmark_restore")
    downloads_dir = root / "downloads"
    extracted_dir = root / "extracted"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for resource in BENCHMARK_ARCHIVES.values():
        zip_path = downloads_dir / resource["zip_name"]
        download_file(resource["url"], zip_path, force=force)

        unpack_dir = extracted_dir / zip_path.stem
        if unpack_dir.exists() and force:
            shutil.rmtree(unpack_dir)
        if not unpack_dir.exists():
            unpack_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(unpack_dir)
            print(f"[extract] {zip_path.name} -> {unpack_dir}")
        else:
            print(f"[skip] extracted {unpack_dir} already exists")

        data_root = _find_tree_root_with_subdirs(unpack_dir, resource["required_subdirs"])
        _symlink_or_copy(data_root, resource["expected_link"], force=force)


def validate_expected_paths(workspace: Path) -> None:
    required = [
        workspace / "examples" / "exports" / "raw" / "94p07mid.sgy",
        workspace / "examples" / "exports" / "raw" / "01p07mid.sgy",
        workspace / "examples" / "exports" / "raw" / "04p07mid.sgy",
        workspace / "examples" / "exports" / "raw" / "06p07mid.sgy",
        workspace / "examples" / "exports" / "raw" / "94p10mid.sgy",
        workspace / "examples" / "exports" / "raw" / "10p10mid.sgy",
        Path("/tmp/sleipner_benchmark_restore/velocities_trends_surfaces/data"),
        Path("/tmp/sleipner_benchmark_restore/Sleipner_Plumes_Boundaries/data"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required prepared assets:\n" + "\n".join(missing))
    print("[ok] all required Sleipner paths are ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare public Sleipner assets for Colab runs.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Path to workspace root (defaults to current working directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract assets even if existing files are present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    if not (workspace / "src" / "ccs_monitoring").exists():
        raise FileNotFoundError(f"{workspace} does not look like the repo root (missing src/ccs_monitoring).")

    prepare_seismic_exports(workspace, force=args.force)
    prepare_benchmark_archives(force=args.force)
    validate_expected_paths(workspace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
