#!/bin/bash
#SBATCH --job-name="test_frame_10s"
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=gpuH200x8
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 6:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out


source venv-gemma/bin/activate
# Exit immediately if a command exits with a non-zero status
set -e

# Define data path
DATA_PATH="nur_json_mix_LB_only"
WAV_PATH="nur_json_mix_LB_only"
TEST_JSON=$DATA_PATH/test_10s_real.json

# Define your paths and parameters here
MODE="frame" # choices from ["event", "frame", "frame_spk_emb"] 
FRAME_SIZE=0.1 # choices from [0.1, 2.0], for frame-based training only 
MODEL_SAVE_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model"
LOG_PATH="Wav2vec-LL-CNN-tinyllama"
MODEL_CONFIG="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/model_config/frame_LB_LLM_0.1s.yaml" # change this accordingly
AUDIO_MODEL_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/whisper_path"
AUDIO_ENCODER_NAME="openai/whisper-large-v2"


LLM_MODEL_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/llm_path"
CUDA_LAUNCH_BLOCKING=1

BATCH_SIZE=32
NUM_WORKERS=4
MAX_EPOCHS=1

# Stage 3: Evaluation
# Specify best checkpoint path, if any
BEST_CHECKPOINT_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model/checkpoints/whisper_1b_frame_10s_3rd_stage9_.ckpt"
evaluate_model() {
    echo "===== Stage 3: Model Evaluation ====="
    # Example command to run the testing script
    torchrun --nproc_per_node=1 \
    test_LB_frame.py \
        --mode "$MODE" \
        --frame_size "$FRAME_SIZE" \
        --model_config "$MODEL_CONFIG" \
        --model_save_path "$MODEL_SAVE_PATH" \
        --best_checkpoint_path "$BEST_CHECKPOINT_PATH" \
        --audio_model_path "$AUDIO_MODEL_PATH" \
        --audio_encoder_name "$AUDIO_ENCODER_NAME" \
        --speaker_encoder_name "$SPEAKER_ENCODER_NAME" \
        --speech_enroll_prefix "$SPEECH_ENROLL_PREFIX" \
        --llm_model_path "$LLM_MODEL_PATH" \
        --log_path "$LOG_PATH" \
        --test_json "$TEST_JSON" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" 
}

# # Execute the pipeline stages
# # prepare_data
# train_model
evaluate_model

echo "Pipeline completed successfully!"
