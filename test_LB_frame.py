import argparse
import torch
import yaml
import os
from datetime import timedelta
import torch.utils.data as data_utils

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
import glob
from load_data import load_testing_data
from load_model import load_testing_model, load_training_model
from LB_callback import MetricCheckpointCallback

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for LB-LLM Models")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["event", "frame", "frame_spk_emb"], required=True,
                        help="Training mode: event-based, frame-based, or frame-based with speaker embeddings.")
    parser.add_argument("--frame_size", type=float, default=2.0, choices=[0.1, 2.0], 
                        help="Frame size in seconds (for frame-based training).")
    # Training configuration
    parser.add_argument("--model_config", type=str, required=True, help="Path to save the model")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--log_path", type=str, required=True, help="Path for logging")
    parser.add_argument("--best_checkpoint_path", type=str, required=False, default=None, help="Path to best checkpoint of the model, if any")
    parser.add_argument("--audio_model_path", type=str, required=False, default=None, help="Path to best audio model, if any")
    parser.add_argument("--audio_encoder_name", type=str, default="wav2vec-LL4300", choices=["wav2vec-LL4300", "openai/whisper-large-v2"], required=True, help="Path to audio model checkpoint")
    parser.add_argument("--speaker_encoder_name", type=str, default=None, required=False, help="Path to speaker audio model checkpoint")
    parser.add_argument("--speech_enroll_prefix", type=str, default="/work/hdd/bebr/PRJ_LLM_SP25/data/wav_enrollment", required=False, help="Path to audio model checkpoint")
    parser.add_argument("--llm_model_path", type=str, required=False, default=None, help="Path to llm model, if any")
    parser.add_argument("--ref_rttm_prefix", type=str, required=False, help="Path to reference rttm file, only use for diarization error rate evaluation")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of workers for DataLoader")
    parser.add_argument('--inference_only', action='store_true', help="Whether to run inference jobs only without ground-truth labels")
    args = parser.parse_args()
    config = load_config(args.model_config)
    config['audio_encoder_name'] = args.audio_encoder_name
    config['audio_model_path'] = args.audio_model_path
    config['llm_model_path'] = args.llm_model_path
    # config['speaker_encoder_name'] = args.speaker_encoder_name
    # config['speech_enroll_prefix'] = args.speech_enroll_prefix

    print(args)
    # Get model
    if args.best_checkpoint_path:
        best_checkpoint_path=args.best_checkpoint_path
    else: # search for best checkpoint path
        try:
            best_checkpoint_path=glob.glob(os.path.join(args.model_save_path, "checkpoints", "*best_checkpoint*"))[0]
        except:
            raise ValueError('No best checkpoint found!')
    print('best_checkpoint_path: ', best_checkpoint_path)
    model = load_testing_model(args, best_checkpoint_path, config).to('cuda')


    # Load data
 
    tokenizer = model.llm_tokenizer
    test_loader = load_testing_data(args, tokenizer)

    
    # set up trainer
    logger = CSVLogger(args.model_save_path, name=args.log_path)
    
    model.test_file_name=args.test_json
    model.ref_rttm_prefix=args.ref_rttm_prefix
    LB_callback_custom = MetricCheckpointCallback(monitor="val_f1_score", mode="max", save_dir=f"{args.model_save_path}/checkpoints", \
                        filename_prefix=args.log_path, log_interval=timedelta(minutes=10))

    # Test
    trainer = Trainer(
        accelerator='gpu', 
        devices=1,
    #    limit_test_batches = 0.01, 
        # strategy=DDPStrategy(timeout=timedelta(seconds=7200)),
        # callbacks=[LB_callback_custom],
        logger = logger,
    )
    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    main()

