"""Download immutable resources and prepare local, independently loadable data."""

import argparse
import fnmatch
import tempfile
from pathlib import Path

from setup_common import configure_paths, now, resource_pins, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Check cached snapshots only")
    args = parser.parse_args()
    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    paths = configure_paths()
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import snapshot_download

    pins = resource_pins()
    model = pins["model"]
    print(f"Loading pinned model snapshot {model['repo_id']} @ {model['revision']}", flush=True)
    model_path = snapshot_download(
        model["repo_id"], revision=model["revision"],
        cache_dir=str(paths["HF_HUB_CACHE"]), local_files_only=args.offline,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        max_workers=4,
    )
    manifest = {
        "created_utc": now(), "purpose": "development environment resource preparation",
        "offline": args.offline, "model": {**model, "snapshot_path": model_path}, "datasets": [],
    }
    for entry in pins["datasets"]:
        patterns = list(entry["files"].values())
        print(f"Loading pinned dataset {entry['repo_id']} @ {entry['revision']}", flush=True)
        snapshot = Path(snapshot_download(
            entry["repo_id"], repo_type="dataset", revision=entry["revision"],
            cache_dir=str(paths["HF_HUB_CACHE"]), local_files_only=args.offline,
            allow_patterns=patterns + ["README.md", "dataset_infos.json"], max_workers=4,
        ))
        recorded = {**entry, "snapshot_path": str(snapshot), "splits": {}}
        for split, pattern in entry["files"].items():
            files = sorted(str(p) for p in snapshot.rglob("*.parquet")
                           if fnmatch.fnmatch(str(p.relative_to(snapshot)), pattern))
            if not files:
                raise FileNotFoundError(f"No cached parquet files for {entry['repo_id']}/{split}")
            # Local source files also permit preparation with network access disabled.
            data = load_dataset("parquet", data_files={split: files}, split=split,
                                cache_dir=str(paths["HF_DATASETS_CACHE"] / "parquet"))
            target = (paths["HF_DATASETS_CACHE"] / "prepared" / entry["key"] /
                      entry["revision"] / split)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=split + "-", dir=target.parent))
                data.save_to_disk(str(staging))
                staging.rename(target)
            existing = load_from_disk(str(target))
            if len(existing) != len(data) or existing[0] != data[0]:
                raise ValueError(f"Prepared dataset differs from pinned source: {target}")
            recorded["splits"][split] = {
                "rows": len(data), "columns": data.column_names,
                "parquet_files": files, "prepared_path": str(target),
                "fingerprint": data._fingerprint,
            }
            print(f"Prepared {entry['key']}/{split}: {len(data)} rows", flush=True)
        manifest["datasets"].append(recorded)
    output = paths["reports"] / ("resources-offline.json" if args.offline else "resources.json")
    write_json(output, manifest)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
