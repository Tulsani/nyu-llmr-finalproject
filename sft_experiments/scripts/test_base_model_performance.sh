#!/bin/bash
#SBATCH --job-name=pit-base-eval
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
BASE_MODEL="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"

NOISE_COT_TEST="$REPO_DIR/dataset/noise_cot_test.jsonl"
GSM8K_TEST="$REPO_DIR/usable_dataset/gsm8k_processed_test.json"

RESULTS_DIR="$REPO_DIR/eval_results_noise_cot"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$RESULTS_DIR"

echo "=============================="
echo "Base Model Evaluation"
echo "=============================="
echo "Base model: $BASE_MODEL"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    # ── [1/2] Base model — noise_cot_test ─────────────────────────────────────
    echo ''
    echo '=== [1/2] Base model — noise_cot_test ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $BASE_MODEL \
      --eval_file   $NOISE_COT_TEST \
      --batch_size  16 \
      --max_new_tokens 1024 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/base_noise_cot_test.json \
      --use_vllm \
      --use_chat_template

    # ── [2/2] Base model — GSM8K ──────────────────────────────────────────────
    echo ''
    echo '=== [2/2] Base model — GSM8K ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $BASE_MODEL \
      --eval_file   $GSM8K_TEST \
      --batch_size  16 \
      --max_new_tokens 1024 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/base_gsm8k.json \
      --use_vllm \
      --use_chat_template

    echo ''
    echo '=============================='
    echo 'All results saved to $RESULTS_DIR'
    echo '=============================='
  "
