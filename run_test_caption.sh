#!/bin/bash
#SBATCH --job-name="test_event_merge"
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=gpuH200x8
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 4:00:00
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
TEST_JSON="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/json_event_based/test_snr_10_mix_8_5s.json"


# Define your paths and parameters here
MODE="event" # choices from ["event", "frame", "frame_spk_emb"] 
FRAME_SIZE=2.0 # choices from [0.1, 2.0], for frame-based training only 
MODEL_SAVE_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model"
LOG_PATH="Wav2vec-LL-CNN-tinyllama"
MODEL_CONFIG="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/model_config/caption_merge.yaml" # change this accordingly
AUDIO_MODEL_PATH1="/work/hdd/bebr/Models/MODEL_WEIGHTS/wav2vec_LL4300.pt"
AUDIO_ENCODER_NAME1="wav2vec-LL4300"
AUDIO_MODEL_PATH2="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/whisper_path"
AUDIO_ENCODER_NAME2="openai/whisper-large-v2"


LLM_MODEL_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/llm_path"
CUDA_LAUNCH_BLOCKING=1

BATCH_SIZE=32
NUM_WORKERS=4
MAX_EPOCHS=1

# Stage 3: Evaluation
# Specify best checkpoint path, if any
BEST_CHECKPOINT_PATH="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model/checkpoints/event_1b_merge_30s_new_1st_stage8_.ckpt"
evaluate_model() {
    echo "===== Stage 3: Model Evaluation ====="
    # Example command to run the testing script
    # torchrun --nproc_per_node=1 \
    python3 \
    test_LB.py \
        --mode "$MODE" \
        --frame_size "$FRAME_SIZE" \
        --model_config "$MODEL_CONFIG" \
        --model_save_path "$MODEL_SAVE_PATH" \
        --best_checkpoint_path "$BEST_CHECKPOINT_PATH" \
        --audio_model_path1 "$AUDIO_MODEL_PATH1" \
        --audio_encoder_name1 "$AUDIO_ENCODER_NAME1" \
        --audio_model_path2 "$AUDIO_MODEL_PATH2" \
        --audio_encoder_name2 "$AUDIO_ENCODER_NAME2" \
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
