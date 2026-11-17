"""Put a leg's reused inputs in place, each file checked against its hash first, staged under <inputs>/ with
<inputs>/manifest.json ({relative path: sha256}). The second study's (PERSON.md, scripts/stage_person.sh) are the first
study's control teacher completions (teachers/control.jsonl) and control students (adapters/control-s<k>/); the third
study's (CONTRAST.md, scripts/stage_contrast.sh) are the Ephraiem and control teachers' completions.

    python leg/install_inputs.py inputs results
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path


def main(src, dst):
    src, dst = Path(src), Path(dst)
    manifest = json.loads((src / "manifest.json").read_text())
    for rel, want in manifest.items():                 # every file checked before any is placed
        got = hashlib.sha256((src / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{src / rel} is not the staged file: {got[:12]}... != {want[:12]}...")
    for rel in manifest:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if (dst / rel).exists():
            raise SystemExit(f"{dst / rel} already exists; refusing to overwrite it")
        shutil.copyfile(src / rel, dst / rel)
    print(f"installed {len(manifest)} files: {sorted({Path(r).parent.as_posix() for r in manifest})}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
