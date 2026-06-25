# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

set -euo pipefail

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd "$(dirname "$0")/../.."

CONFIG_PATH="${CONFIG_PATH:-config/nskt/unet.yaml}"

python train_unet.py --config "${CONFIG_PATH}"
