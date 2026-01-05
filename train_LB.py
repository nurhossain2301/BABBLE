import argparse
import torch
import yaml
import os
from datetime import timedelta
import torch.utils.data as data_utils
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
# from lightning.pytorch.callbacks import DeviceStatsMonitor

from LB_callback import MetricCheckpointCallback
from load_data import load_training_data
from load_model import load_training_model

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
    parser.add_argument("--audio_model_path1", type=str, required=True, help="Path to audio model checkpoint")
    parser.add_argument("--audio_encoder_name1", type=str, default="wav2vec-LL4300", choices=["wav2vec-LL4300", "openai/whisper-large-v2"], required=True, help="Path to audio model checkpoint")
    parser.add_argument("--audio_model_path2", type=str, required=True, help="Path to audio model checkpoint")
    parser.add_argument("--audio_encoder_name2", type=str, default="openai/whisper-large-v2", choices=["wav2vec-LL4300", "openai/whisper-large-v2"], required=True, help="Path to audio model checkpoint")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to LLM model checkpoint")
    parser.add_argument("--train_json", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--val_json", type=str, required=True, help="Path to validation JSON file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of workers for DataLoader")
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum number of training epochs")
    parser.add_argument('--inference_only', default='False', action='store_true', help="Whether to run inference jobs only without ground-truth labels")
    args = parser.parse_args()

    config = load_config(args.model_config)
    config['audio_encoder_name1'] = args.audio_encoder_name1
    config['audio_model_path1'] = args.audio_model_path1
    config['audio_encoder_name2'] = args.audio_encoder_name2
    config['audio_model_path2'] = args.audio_model_path2
    config['llm_model_path'] = args.llm_model_path


    print(config)
    # Get model
    model = load_training_model(args, config)
    

    # Load data
    tokenizer = model.llm_tokenizer
    train_loader, val_loader = load_training_data(args, tokenizer)
    
    
    # set up trainer
    LB_callback_custom = MetricCheckpointCallback(monitor="val_f1_score", mode="max", save_dir=f"{args.model_save_path}/checkpoints", \
                        filename_prefix=args.log_path, log_interval=timedelta(minutes=15))

    # if os.path.exists(os.path.join(args.model_save_path, "checkpoints", "last.ckpt")):
    # last_checkpoint_path=os.path.join(args.model_save_path, "checkpoints", "last.ckpt")
    # last_checkpoint_path = "/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/logger/Wav2vec-LL-CNN-tinyllama/version_5/checkpoints/epoch=7-step=18668.ckpt"
    last_checkpoint_path = None
    
    logger = TensorBoardLogger(name=args.log_path, save_dir="/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/logger")
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=3600))
    
    #train
    trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='gpu', 
            devices=4, 
            strategy=strategy, 
            # limit_train_batches=0.001,
            limit_val_batches=0,
            log_every_n_steps=100, 
            precision="bf16",
            enable_checkpointing=True, 
            callbacks=[LB_callback_custom],
            fast_dev_run=False, 
            logger=logger, 
            accumulate_grad_batches=config['grad_accumulate_steps'],
            num_sanity_val_steps=0,
            enable_progress_bar=True
    )
    # print('here')
    # trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    trainer.fit(model, train_loader, val_loader, ckpt_path = "/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/saved_model/checkpoints/frame_1b_merge_30s_new_1st_stage5_.ckpt")
    # trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
