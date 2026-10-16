#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export UV_LINK_MODE=copy UV_CACHE_DIR=/workspace/.uvcache HF_HOME=/workspace/hf LC_ALL=C.UTF-8
mkdir -p logs runs
git init -q villa && cd villa && git remote add origin https://github.com/ScrollPrize/villa.git
git sparse-checkout set vesuvius && git fetch -q --depth 1 --filter=blob:limit=5m origin ccb431818fa68059138f2c964ae299fc474ff116 && git checkout -q FETCH_HEAD
git log -1 --format='villa %H %cd'
cd /workspace
uv venv -q -p 3.14 /workspace/venv
. /workspace/venv/bin/activate
uv pip install -q "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q numpy scipy "zarr>=2.18.7,<4" "numcodecs>=0.15.1,<0.17" s3fs fsspec huggingface_hub matplotlib numba pillow requests pyyaml cachetools edt aiohttp tifffile pandas scikit-learn pynrrd opencv-python-headless
uv pip install -q --no-deps -e villa/vesuvius
python -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
python -c "from huggingface_hub import hf_hub_download as d; print(d('scrollprize/fiber_single_small','last.pt',revision='451182558cb3270cfc2ddf967acbd4f36b9b816b'))"
python code/fetch_fibers.py /workspace/fibers
for d in fibers/*; do echo "$d $(ls $d | wc -l)"; done
echo SETUP_DONE
