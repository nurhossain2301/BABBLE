#!/bin/bash
#SBATCH --job-name="train_audio"
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=gpuA40x4,gpuA100x8
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 12:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=mkhan@wpi.edu    # <- replace "NetID" with your University NetID
#SBATCH --mail-type=BEGIN,END

source venv-gemma/bin/activate
# Exit immediately if a command exits with a non-zero status
set -e

# Define data path
DATA_PATH="nur_json_mix_LB_only"
WAV_PATH="nur_json_mix_LB_only"
# TRAIN_JSON=$DATA_PATH/train_LB_LENA_voc_snr_5_mix_8_skip3.json
TRAIN_JSON=$DATA_PATH/train_2s_real.json
# VAL_JSON=$DATA_PATH/dev_snr_10_mix_8.json
VAL_JSON=$DATA_PATH/dev_2s_real.json
TEST_JSON=$DATA_PATH/test_2s_real.json

# Define your paths and parameters here
MODE="frame" # choices from ["event", "frame", "frame_spk_emb"] 
FRAME_SIZE=0.1 # choices from [0.1, 2.0], for frame-based training only 
MODEL_SAVE_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model"
LOG_PATH="Wav2vec-LL-CNN-tinyllama"
MODEL_CONFIG="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/model_config/frame_LB_LLM_0.1s.yaml" # change this accordingly
AUDIO_MODEL_PATH="/work/hdd/bebr/Models/MODEL_WEIGHTS/wav2vec_LL4300.pt"
AUDIO_ENCODER_NAME="wav2vec-LL4300"


LLM_MODEL_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/llm_path"
CUDA_LAUNCH_BLOCKING=1
BEST_CHECKPOINT_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/logger/Wav2vec-LL-CNN-tinyllama/version_5/checkpoints/epoch=7-step=18668.ckpt"

BATCH_SIZE=64
NUM_WORKERS=4
MAX_EPOCHS=20

train_model() {
    echo "===== Stage 2: Model Training ====="
    # Example command to run the training script
    # accelerate launch --num_processes 2 \
    torchrun --nproc_per_node=1 \
    train_audio_encoder.py \
            --mode "$MODE" \
            --frame_size "$FRAME_SIZE" \
            --model_config "$MODEL_CONFIG" \
            --model_save_path "$MODEL_SAVE_PATH" \
            --best_checkpoint_path "$BEST_CHECKPOINT_PATH" \
            --log_path "$LOG_PATH" \
            --audio_model_path "$AUDIO_MODEL_PATH" \
            --audio_encoder_name "$AUDIO_ENCODER_NAME" \
            --llm_model_path "$LLM_MODEL_PATH" \
            --train_json "$TRAIN_JSON" \
            --val_json "$VAL_JSON" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --max_epochs "$MAX_EPOCHS"
}

train_model

echo "Pipeline completed successfully!"