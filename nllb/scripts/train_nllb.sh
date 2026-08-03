#!/bin/bash
# Fine-tune extended NLLB model on gloss-to-translation data
# Adapted from PARME: https://github.com/DOLMA-NLP/PARME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
MODEL_DIR="${PROJECT_ROOT}/models/nllb_extended"
DATA_DIR="${PROJECT_ROOT}/data"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
VAL_FILE="${DATA_DIR}/val.jsonl"
OUTPUT_DIR="${NLLB_OUTPUT_DIR:-${PROJECT_ROOT}/models/nllb_finetuned}"

# Training hyperparameters
BATCH_SIZE=2
GRADIENT_ACCUM=16
LEARNING_RATE=2e-5
NUM_EPOCHS=10
MAX_SOURCE_LENGTH=128
MAX_TARGET_LENGTH=128
EVAL_STEPS=500
SAVE_STEPS=500

mkdir -p "${OUTPUT_DIR}"

echo "=== GPU preflight check ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. A CUDA GPU is required for training."
    exit 1
fi

if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed. Ensure a GPU is available to this process."
    exit 1
fi
nvidia-smi -L

python - <<'EOF'
import sys
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device:    {torch.cuda.get_device_name(0)}")
    print(f"GPU count:     {torch.cuda.device_count()}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU memory:    {mem_gb:.1f} GB")
else:
    print("\nERROR: PyTorch cannot see a GPU.")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo "=== GPU check passed, starting training ==="
echo "Output directory: ${OUTPUT_DIR}"

python "${SCRIPT_DIR}/run_translation.py" \
    --model_name_or_path "${MODEL_DIR}" \
    --do_train \
    --source_lang gls_Latn \
    --target_lang eng_Latn \
    --forced_bos_token eng_Latn \
    --train_file "${TRAIN_FILE}" \
    --validation_file "${VAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --max_target_length ${MAX_TARGET_LENGTH} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --predict_with_generate \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --overwrite_cache

echo "Training completed! Model saved to ${OUTPUT_DIR}"
