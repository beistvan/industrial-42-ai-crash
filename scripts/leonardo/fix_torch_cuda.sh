#!/usr/bin/env bash
# Reinstall PyTorch for Leonardo A100 (CUDA 12.1 driver). Run on login node.
#
# Problem: pip install -r requirements.txt pulls torch+cu130, but Leonardo
# nodes have CUDA 12.1/12.2 drivers → torch.cuda.is_available() is false.
#
# Usage:
#   cd $SCRATCH/industrial
#   bash scripts/leonardo/fix_torch_cuda.sh

set -euo pipefail

cd "${SCRATCH:?}/industrial"

if [[ -f scripts/leonardo/load_env.sh ]]; then
    # shellcheck source=scripts/leonardo/load_env.sh
    source scripts/leonardo/load_env.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Current torch:"
python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "(not installed)"

echo "Reinstalling torch for CUDA 12.1 (Leonardo A100)..."
pip uninstall -y torch triton 2>/dev/null || true
pip uninstall -y nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 \
    nvidia-cuda-runtime-cu13 nvidia-cudnn-cu13 nvidia-cufft-cu13 nvidia-cufile-cu13 \
    nvidia-curand-cu13 nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 nvidia-nvtx-cu13 \
    cuda-bindings cuda-toolkit cuda-pathfinder 2>/dev/null || true

pip install torch --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installed: $(python -c 'import torch; print(torch.__version__)')"
echo "cuda.is_available() on login node (may be False — normal):"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
echo ""
echo "Submit a 1-min GPU smoke to verify on a compute node:"
echo '  sbatch --account=EUHPC_D30_031 --partition=boost_usr_prod --reservation=s_tra_ncc \'
echo '    --nodes=1 --ntasks=1 --gpus=1 --time=00:05:00 --chdir=$SCRATCH/industrial \'
echo '    --output=logs/slurm-torch-smoke-%j.out \'
echo '    --wrap "module load cuda/12.1 2>/dev/null || true; source .venv/bin/activate && python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\""'
