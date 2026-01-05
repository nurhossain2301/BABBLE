from trainer import EventTrainer, FrameSpkEmbTrainer, FrameTrainer
# from trainer_frame import FrameTrainer

def load_training_model(args, model_config):
    if args.mode=="event":
        return EventTrainer(**model_config)
    elif args.mode=="frame":
        return FrameTrainer(**model_config)
    else: #frame_spk_emb
        return FrameSpkEmbTrainer(**model_config)

def load_testing_model(args, best_checkpoint_path, model_config):
    if args.mode=="event":
        model = EventTrainer.load_from_checkpoint(best_checkpoint_path, **model_config)
    elif args.mode=="frame":
        model = FrameTrainer.load_from_checkpoint(best_checkpoint_path,  **model_config)
    else: #frame_spk_emb
        model = FrameSpkEmbTrainer.load_from_checkpoint(best_checkpoint_path,  **model_config)
    return model
