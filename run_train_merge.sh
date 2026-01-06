#!/bin/bash
#SBATCH --job-name="train_frame_merge"
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=gpuA100x8,gpuA40x4
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=4
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 12:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out


source venv-gemma/bin/activate
# Exit immediately if a command exits with a non-zero status
set -e

# Define data path
DATA_PATH="nur_json_mix_LB_only"
WAV_PATH="nur_json_mix_LB_only"
# TRAIN_JSON=$DATA_PATH/train_LB_LENA_voc_snr_5_mix_8_skip3.json
# TRAIN_JSON=$DATA_PATH/train_10s_real.json
# VAL_JSON=$DATA_PATH/test_10s_real.json
# TEST_JSON=$DATA_PATH/test_10s_real.json
TRAIN_JSON="/work/hdd/bebr/PRJ_LLM_SP25/data/nur_json_file/train_30s_real.json"
VAL_JSON="/work/hdd/bebr/PRJ_LLM_SP25/data/nur_json_file/test_30s_real.json"
TEST_JSON="/work/hdd/bebr/PRJ_LLM_SP25/data/nur_json_file/test_30s_real.json"
# Define your paths and parameters here
MODE="frame" # choices from ["event", "frame", "frame_spk_emb"] 
FRAME_SIZE=0.1 # choices from [0.1, 2.0], for frame-based training only 
MODEL_SAVE_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model"
LOG_PATH="Wav2vec-LL-CNN-tinyllama"
MODEL_CONFIG="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/model_config/merge_0.1s.yaml" # change this accordingly
AUDIO_MODEL_PATH1="/work/hdd/bebr/Models/MODEL_WEIGHTS/wav2vec_LL4300.pt"
AUDIO_MODEL_PATH2="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/whisper_path"
AUDIO_ENCODER_NAME2="openai/whisper-large-v2"
AUDIO_ENCODER_NAME1="wav2vec-LL4300"


LLM_MODEL_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/llm_path"
CUDA_LAUNCH_BLOCKING=1

BATCH_SIZE=16
NUM_WORKERS=4
MAX_EPOCHS=9

train_model() {
    echo "===== Stage 2: Model Training ====="
    # Example command to run the training script
    # accelerate launch --num_processes 2 \
    torchrun --nproc_per_node=4 \
        train_LB.py \
            --mode "$MODE" \
            --frame_size "$FRAME_SIZE" \
            --model_config "$MODEL_CONFIG" \
            --model_save_path "$MODEL_SAVE_PATH" \
            --log_path "$LOG_PATH" \
            --audio_model_path1 "$AUDIO_MODEL_PATH1" \
            --audio_encoder_name1 "$AUDIO_ENCODER_NAME1" \
            --audio_model_path2 "$AUDIO_MODEL_PATH2" \
            --audio_encoder_name2 "$AUDIO_ENCODER_NAME2" \
            --llm_model_path "$LLM_MODEL_PATH" \
            --train_json "$TRAIN_JSON" \
            --val_json "$VAL_JSON" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --max_epochs "$MAX_EPOCHS"
}

train_model

echo "Pipeline completed successfully!"