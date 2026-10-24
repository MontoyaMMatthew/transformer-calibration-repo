#!/usr/bin/env bash
# Run once on the recorded RunPod PyTorch 2.8.0/CUDA 12.8 template.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
python - <<'PY'
import sys, torch
from pathlib import Path
from scripts.setup_common import persistent_root
if sys.version_info[:2] != (3, 12) or torch.__version__ != '2.8.0+cu128':
    raise RuntimeError('Requires the recorded Python 3.12 / torch 2.8.0+cu128 RunPod base; validate other templates separately')
if not Path.cwd().resolve().is_relative_to(persistent_root()):
    raise RuntimeError('Place the project inside TRANSFORMER_CALIBRATION_PERSISTENT_ROOT')
PY
python -m venv --system-site-packages .venv
mkdir -p cache/tmp artifacts/environment
python - <<'PY'
from pathlib import Path
from scripts.setup_common import persistent_root
root = Path.cwd()
p = root / '.env'
if not p.exists():
    values = {
        'HF_HOME': root/'cache/huggingface', 'HF_HUB_CACHE': root/'cache/huggingface/hub',
        'HF_DATASETS_CACHE': root/'cache/huggingface/datasets',
        'TRANSFORMER_CALIBRATION_ARTIFACTS': root/'artifacts',
        'MPLCONFIGDIR': root/'cache/matplotlib', 'PIP_CACHE_DIR': root/'cache/pip',
        'TMPDIR': root/'cache/tmp', 'XDG_CACHE_HOME': root/'cache/xdg',
        'TORCH_HOME': root/'cache/torch', 'TRITON_CACHE_DIR': root/'cache/triton',
        'CUDA_CACHE_PATH': root/'cache/cuda',
        'TRANSFORMER_CALIBRATION_PERSISTENT_ROOT': persistent_root(),
        'TRANSFORMER_CALIBRATION_TORCH_SOURCE': 'RunPod-template-torch-2.8.0+cu128',
        'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_DISABLE_XET': '1', 'HF_HUB_ENABLE_HF_TRANSFER': '0',
    }
    p.write_text(''.join(f'{k}={v}\n' for k, v in values.items()))
PY
set -a
source .env
set +a
if [[ -f requirements.runpod-overlay.txt ]]; then
    .venv/bin/python -m pip install -r requirements.runpod-overlay.txt
else
    .venv/bin/python -m pip install -r requirements.runpod.in
fi
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > artifacts/environment/runpod-packages.txt
echo 'Environment ready. Use scripts/runpod for preparation and validation. Pilot has not started.'
