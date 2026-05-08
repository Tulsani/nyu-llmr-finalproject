#!/bin/bash
#SBATCH --job-name=pit-noise-cot-v2v3-test
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
V1_MODEL="$REPO_DIR/checkpoints/pit_noise_cot_noise_all_V2/checkpoint-200"
V2_MODEL="$REPO_DIR/checkpoints/pit_noise_cot_noise_all_V3/checkpoint-200"

TEST_DATA="$REPO_DIR/dataset/test_data.jsonl"

RESULTS_DIR="$REPO_DIR/eval_results_noise_cot"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$RESULTS_DIR"

echo "=============================="
echo "Noise-CoT V1 vs V2 — test_data Eval"
echo "=============================="
echo "V1 model: $V1_MODEL"
echo "V2 model: $V2_MODEL"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    # ── [1/2] V1 model (pit_noise_cot_noise_all_V2) — test_data ───────────────
    echo ''
    echo '=== [1/2] V1 model — test_data ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $V1_MODEL \
      --eval_file   $TEST_DATA \
      --batch_size  16 \
      --max_new_tokens 1024 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/v1_model_test_data.json \
      --use_vllm

    # ── [2/2] V2 model (pit_noise_cot_noise_all_V3) — test_data ───────────────
    echo ''
    echo '=== [2/2] V2 model — test_data ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $V2_MODEL \
      --eval_file   $TEST_DATA \
      --batch_size  16 \
      --max_new_tokens 1024 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/v2_model_test_data.json \
      --use_vllm

    echo ''
    echo '=============================='
    echo 'All results saved to $RESULTS_DIR'
    echo '=============================='
  "
