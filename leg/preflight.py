# Edited for release.
"""Fail in seconds, before any download: the venv is clean, CUDA works, bf16 is there, the GPU is big enough, the
pinned versions are the ones installed, and the inputs are the ones built locally (data/inputs.json)."""
import hashlib, importlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

pins = {l.split("==")[0]: l.split("==")[1].strip() for l in (ROOT / "leg" / "requirements.txt").read_text().splitlines()
        if "==" in l and not l.startswith("#")}
for pkg in ("vllm", "torch", "transformers", "peft", "accelerate", "numpy"):
    mod = importlib.import_module(pkg)
    got = mod.__version__.split("+")[0]
    assert got == pins[pkg], f"{pkg} {got} is not the pinned {pins[pkg]}"
    assert str(ROOT / "rvenv") in mod.__file__, f"{pkg} is imported from outside the venv: {mod.__file__}"
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.is_bf16_supported(), "no bf16"
gb = torch.cuda.get_device_properties(0).total_memory / 2**30
assert gb >= 38, f"the GPU has {gb:.0f} GB; the leg needs a 40 GB card or larger"
# the first study's inputs always; the second study's (PERSON.md) when its leg asks: `preflight.py person`
for manifest in ("inputs.json", *(["person_inputs.json"] if "person" in sys.argv[1:] else [])):
    for name, want in json.loads((ROOT / "data" / manifest).read_text())["sha256"].items():
        assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == want, f"data/{name} changed"
from transformers import Qwen2ForCausalLM  # noqa: E402,F401  (the class the leg loads; a broken install fails here)
from susurrus import config, data, person  # noqa: E402,F401
print(f"PREFLIGHT_OK torch {torch.__version__} on {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")
