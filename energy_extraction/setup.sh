# 0) New venv
uv venv .venv
source .venv/bin/activate

# 1) Core libs (pin stable wheels)
uv pip install "pymupdf==1.24.9" "numpy==1.26.4" "opencv-python==4.10.0.84" "layoutparser==0.3.4"

# 2) PyTorch CPU
uv pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu

# 3) Detectron2 from source, but allow it to see the torch you just installed
#    (builds a CPU-only wheel; avoids the build-isolation 'no torch' error)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# 4) Sanity checks
python - <<'PY'
import torch, detectron2
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("detectron2 OK")
PY
