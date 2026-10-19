"""Checksummed portable research bundle; copy it independently off the Pod."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", type=Path, nargs="+", required=True,
                        help="Run, analysis and/or benchmark directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to("/workspace") or output.exists():
        raise ValueError("Choose a new archive path under /workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {}
    with tarfile.open(output, "w:gz") as archive:
        for directory in args.include:
            directory = directory.resolve()
            if output.is_relative_to(directory):
                raise ValueError("Archive cannot be inside an included directory")
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.name != "writer.lock" and not path.name.endswith(".tmp"):
                    name = str(Path(directory.name) / path.relative_to(directory))
                    if name in members:
                        raise ValueError(f"Duplicate archive member: {name}")
                    members[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    archive.add(path, arcname=name, recursive=False)
    with tarfile.open(output, "r:gz") as archive:
        for name, expected in members.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise ValueError(f"Archive checksum mismatch: {name}")
    output.with_suffix(output.suffix + ".members.json").write_text(json.dumps(members, indent=2) + "\n")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"Verified {len(members)} members: {output}")


if __name__ == "__main__":
    main()
