#!/bin/bash
#SBATCH --job-name=qwen_lora_5k
#SBATCH --account=project_2014607
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2014607/logs/%x-%j.out
#SBATCH --error=/scratch/project_2014607/logs/%x-%j.err

set -euo pipefail

# --- Paths ---
BASE=/scratch/project_2014607
LOGS="$BASE/logs"
HF_HOME="$BASE/hf_cache"
MPLCONFIGDIR="$BASE/mpl_config"
NLTK_DATA="$BASE/nltk_data"
TMPDIR="$BASE/tmp"
PIP_CACHE_DIR="$BASE/pip_cache"
PYUSER="$BASE/pyuser"
DATA_CSV="$BASE/mimic.csv"          # 4,998 rows
KG_CSV="$BASE/kg/clin_kg.csv"       # used automatically if present

mkdir -p "$LOGS" "$HF_HOME" "$MPLCONFIGDIR" "$NLTK_DATA" "$TMPDIR" "$PIP_CACHE_DIR" "$PYUSER" "$BASE/kg"
cd "$BASE"

# --- Environment / modules ---
module --force purge
module load pytorch

export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export MPLCONFIGDIR
export NLTK_DATA
export TMPDIR
export PIP_CACHE_DIR
export PYTHONUSERBASE="$PYUSER"

# Ensure user-site is on PATH & PYTHONPATH
pyver=$(python - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
export PYTHONPATH="$PYUSER/lib/python${pyver}/site-packages:${PYTHONPATH:-}"
export PATH="$PYUSER/bin:$PATH"

# --- Deps ---
python -m pip install --user --upgrade \
  "transformers>=4.44.0" \
  datasets \
  peft \
  evaluate \
  bert-score \
  nltk \
  matplotlib \
  scikit-learn \
  accelerate \
  rouge-score \
  sentence-transformers

# Sanity print
python - <<'PY'
import torch, sys, os
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| python", sys.version.split()[0])
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("PATH has pyuser/bin:", ("/scratch/project_2014607/pyuser/bin" in os.environ.get("PATH","")))
PY

# --- Training ---
python train_qwen_lora.py \
  --kg_csv "$KG_CSV" \
  --checkpoint Qwen/Qwen2.5-7B-Instruct \
  --data_csv "$DATA_CSV" \
  --subset 0 \
  --epochs 8 \
  --lr 4e-5 \
  --train_bs 2 --eval_bs 1 --grad_accum 16 \
  --max_src_len 512 --max_tgt_len 380 \
  --eval_subset 400 \
  --preds_n 5 \
  --gen_n 50 --gen_beams 6
