# Installation

Use separate environments for geometry and video to avoid incompatible CUDA extension wheels. These instructions pin the packages exercised on the release test server; a clean installation on another GPU has not been timed or benchmarked.

## Geometry

Python 3.10, Linux x86_64, CUDA-compatible driver, PyTorch 2.5.1. The verification GPU was NVIDIA A800 80 GB.

```bash
python3.10 -m venv .venv-geometry
source .venv-geometry/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2+pt25cu121 --no-index --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install cumm-cu121==0.7.11 spconv-cu121==2.3.8
pip install -r environments/requirements.txt
# urdfpy has an obsolete networkx==2.2 dependency pin. Use the tested modern version above.
pip install urdfpy==0.0.22 --no-deps
pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
pip install -e . --no-deps
python -m dexplicit.geometry --help
```

The FlashAttention wheel is specific to CPython 3.10 and the official PyTorch 2.5 binary ABI. Compile the pinned FlashAttention version for a different ABI; do not mix CUDA 11 and CUDA 12 extension wheels.

## Video

The tested renderer environment used PyTorch 2.5.1 + CUDA 11.8. Install the pinned DiffSynth backend in the checkout, then use `--backend third_party/DiffSynth-Studio` in commands.

```bash
python3.10 -m venv .venv-video
source .venv-video/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==2.2.6 safetensors==0.8.0 einops==0.8.2 pillow \
  transformers==5.13.0 accelerate==1.14.0 modelscope==1.38.0 \
  peft==0.9.0 diffusers==0.33.1 imageio imageio-ffmpeg==0.6.0 opencv-python-headless \
  huggingface-hub==1.22.0 sentencepiece==0.2.1
mkdir -p third_party
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
git -C third_party/DiffSynth-Studio checkout ce9f4541d0f4ae47a138337e59066dd633207aa7
pip install -e . --no-deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m dexplicit.infer --help
```

The tested environment also had FlashAttention 2.7.3 for CUDA 11/PyTorch 2.5. The backend can select another attention implementation when unavailable; resulting pixels may differ. Keep the inference backend and CUDA/library versions fixed for numerical comparisons.

For the selected 14B pipeline, reserve an 80 GB GPU and sufficient system RAM for both experts and CPU offloading (at least 128 GB recommended). The backbone files occupy about 81 GB in the model cache. Initial download and model loading are separate from sampling time. The tested configuration uses native CPU offloading, batch size 1 and separate CFG passes.

For offline 14B operation, pass `--base-model-dir /path/to/Wan2.2-VACE-Fun-A14B`; it must contain `high_noise_model/diffusion_pytorch_model.safetensors`, `low_noise_model/diffusion_pytorch_model.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth` and `google/umt5-xxl/`. `--tokenizer` can override the tokenizer directory. The default downloader pins the official upstream Hugging Face revision recorded in the manifest.

The optional compact renderer uses `--base-model /path/to/diffusion_pytorch_model.safetensors` and `--vae /path/to/Wan2.1_VAE.pth`. It requires the compact upstream model, not the 14B files.

Run CPU interface checks with `python -m unittest discover -s tests -v`.
