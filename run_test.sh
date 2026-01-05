# #!/bin/bash
### Example script to run inference only and generate labels without ground-truth labels on LB-LLM

# Exit immediately if a command exits with a non-zero status
set -e

# Define data path 
DATA_PATH="nur_json_mix_LB_only" # path to store json file
WAV_PATH="/work/hdd/bebr/PRJ_LLM_SP25/data/test_wav" # path to wav files

# Define your paths and parameters here
MODE="frame" # choices from ["event", "frame", "frame_spk_emb"] 
FRAME_SIZE=0.1 # choices from [0.1, 2.0], for frame-based training only 
MODEL_SAVE_PATH="/u/mkhan14/LittleBeats-LLM/saved_model"
LOG_PATH="Wav2vec-LL-CNN-tinyllama"
AUDIO_MODEL_PATH="/work/hdd/bebr/MODEL_WEIGHTS/wav2vec-LL4300.pt"
AUDIO_ENCODER_NAME="wav2vec-LL4300"
LLM_MODEL_PATH="/u/mkhan14/LittleBeats-LLM/llm_path" # path where LLM will be downloaded/stored

BATCH_SIZE=1
NUM_WORKERS=2

# Stage 1: Data Preparation
# prepare_data() {
#     echo "===== Stage 1: Data Preparation ====="
#     # Example command to extract json files given a wav file folder
#     python data_prep.py \
#         --folder_path "$WAV_PATH" \
#         --output_dir "$DATA_PATH"
# }
TEST_JSON=$DATA_PATH/test_snr_10_mix_8.json

# Stage 2: Evaluation
# Specify best checkpoint path, if any
BEST_CHECKPOINT_PATH=None
evaluate_model() {
    echo "===== Stage 2: Model Evaluation ====="
    # Example command to run the testing script
    python test_LB.py \
        --mode "$MODE" \
        --frame_size "$FRAME_SIZE" \
        --model_save_path "$MODEL_SAVE_PATH" \
        --log_path "$LOG_PATH" \
        --test_json "$TEST_JSON" \
        --batch_size "$BATCH_SIZE" \
        --best_checkpoint_path "$BEST_CHECKPOINT_PATH" \
        --num_workers "$NUM_WORKERS" \
        --inference_only
}

# Execute the pipeline stages
#prepare_data
evaluate_model

echo "Pipeline completed successfully!"
