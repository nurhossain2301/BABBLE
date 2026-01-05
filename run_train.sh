#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define data path
DATA_PATH="nur_json_mix_LB_only"
WAV_PATH="nur_json_mix_LB_only"
TRAIN_JSON=$DATA_PATH/train_snr_5_mix_8_2s.json
VAL_JSON=$DATA_PATH/test_snr_5_mix_8_2s.json
TEST_JSON=$DATA_PATH/test_snr_5_mix_8_2s.json

# Define your paths and parameters here
MODE="frame_spk_emb" # choices from ["event", "frame", "frame_spk_emb"] 
FRAME_SIZE=2.0 # choices from [0.1, 2.0], for frame-based training only 
MODEL_SAVE_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model"
LOG_PATH="Wav2vec-LL-CNN-tinyllama"
MODEL_CONFIG="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/model_config/event_LB_LLM.yaml" # change this accordingly
AUDIO_MODEL_PATH="/work/hdd/bebr/Models/MODEL_WEIGHTS/wav2vec_LL4300.pt"
AUDIO_ENCODER_NAME="wav2vec-LL4300"
SPEAKER_ENCODER_NAME="wav2vec-LL4300"
SPEECH_ENROLL_PREFIX="/work/hdd/bebr/PRJ_LLM_SP25/data/wav_enrollment"

LLM_MODEL_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/llm_path"
CUDA_LAUNCH_BLOCKING=1

BATCH_SIZE=8
NUM_WORKERS=4
MAX_EPOCHS=1

# Stage 1: Data Preparation
prepare_data() {
    echo "===== Stage 1: Data Preparation ====="
    ### write your own function here to preprocess data
}

# Stage 2: Model Training
train_model() {
    echo "===== Stage 2: Model Training ====="
    # Example command to run the training script
    # accelerate launch --num_processes 2 \
    torchrun --nproc_per_node=1 \
        train_LB.py \
            --mode "$MODE" \
            --frame_size "$FRAME_SIZE" \
            --model_config "$MODEL_CONFIG" \
            --model_save_path "$MODEL_SAVE_PATH" \
            --log_path "$LOG_PATH" \
            --audio_model_path "$AUDIO_MODEL_PATH" \
            --audio_encoder_name "$AUDIO_ENCODER_NAME" \
            --speaker_encoder_name "$SPEAKER_ENCODER_NAME" \
            --speech_enroll_prefix "$SPEECH_ENROLL_PREFIX" \
            --llm_model_path "$LLM_MODEL_PATH" \
            --train_json "$TRAIN_JSON" \
            --val_json "$VAL_JSON" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --max_epochs "$MAX_EPOCHS"
}

# Stage 3: Evaluation
# Specify best checkpoint path, if any
BEST_CHECKPOINT_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model/checkpoints/event_tiny.ckpt"
evaluate_model() {
    echo "===== Stage 3: Model Evaluation ====="
    # Example command to run the testing script
    torchrun --nproc_per_node=1 \
    test_LB.py \
        --mode "$MODE" \
        --frame_size "$FRAME_SIZE" \
        --model_save_path "$MODEL_SAVE_PATH" \
        --best_checkpoint_path "$BEST_CHECKPOINT_PATH" \
        --audio_model_path "$AUDIO_MODEL_PATH" \
        --llm_model_path "$LLM_MODEL_PATH" \
        --log_path "$LOG_PATH" \
        --test_json "$TEST_JSON" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" 
}

# # Execute the pipeline stages
# # prepare_data
train_model
# evaluate_model

echo "Pipeline completed successfully!"
