#!/bin/bash
#SBATCH --job-name=mpac_verify
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /nfs/roberts/scratch/pi_skr2/mcn26/mpac_hf/verify_gpu-%j.out
#SBATCH -e /nfs/roberts/scratch/pi_skr2/mcn26/mpac_hf/verify_gpu-%j.err

set -euo pipefail

ROOT=/nfs/roberts/scratch/pi_skr2/mcn26/mpac_hf
PY=$ROOT/venv/bin/python
TABLE=/nfs/roberts/scratch/pi_skr2/go274/coda_data/DATA-Table_S2__MPRA_dataset.txt

echo "[*] node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-unset}"
$PY -c "import torch; print('[*] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd /nfs/roberts/project/pi_skr2/mcn26/coda_mpac/hf_export

$PY verify.py \
    --export "$ROOT/export" \
    --table_s2 "$TABLE" \
    --batch_size 256 \
    --device cuda

echo "[+] done."
