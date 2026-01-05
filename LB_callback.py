import os
from datetime import datetime, timedelta
from lightning.pytorch.callbacks import Callback
# from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
import pdb
import glob
import json
import time
import copy


# f = open('record_test_batch_callback_text2.txt', 'w')
class MetricCheckpointCallback(Callback):
    """
    Custom callback to log important metrics and save checkpoints based on the monitored metric.
    """

    def __init__(self, monitor="val_loss", mode="min", save_dir="checkpoints", filename_prefix="best_checkpoint",log_interval=None):
        """
        Args:
            monitor (str): The metric to monitor (default is "val_f1_score").
            mode (str): One of {"min", "max"}. In "min" mode, the checkpoint is saved when the metric decreases.
                        In "max" mode, the checkpoint is saved when the metric increases.
            save_dir (str): Directory to save checkpoints.
            filename (str): Name of the checkpoint file.
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        self.all_epochs_metrics = {}
        self.log_interval = log_interval
        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.meta_data_path = os.path.join(self.save_dir, "meta_data.json")
        if os.path.exists(self.meta_data_path):
            f=open(self.meta_data_path,"r")
            self.all_epochs_metrics=json.load(f)
            self.best_score = self.all_epochs_metrics["best_score"]
            f.close()
        self.last_log_time = datetime.now()

    # @rank_zero_only
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     current_time = datetime.now()
    #     if current_time - self.last_log_time >= self.log_interval:
    #         self.last_log_time = current_time
    #         filename = os.path.join(self.save_dir, \
    #             "frame_06_28_tiny_full_v2.ckpt")
    #         self._save_checkpoint(trainer, pl_module, filename)

            # pl_module.eval()
            # # waveform, inputs, targets,instruction_len, data_ids,  = batch
            # waveform, pre_speech_prompt, post_speech_prompt, output_prompt, data_ids=batch
            # # print('hererere', instruction_len)
            
            # embeds, atts_mask, embeds_no_output, atts_mask_no_output, target_ids = pl_module.encode(waveform, pre_speech_prompt, post_speech_prompt, output_prompt,)
            # # print('here')
            # predicted_ids=pl_module.generate_text(embeds_no_output, atts_mask_no_output)
            # # print('heelo')
            # for i in range(len(predicted_ids)):   
            #     # input_ids_after = trainer.llm_tokenizer(message, add_special_tokens=False, return_tensors='pt').input_ids
            #     # target_ids_after = copy.deepcopy(input_ids_after[-1]) 
            #     target_text = pl_module.llm_tokenizer.decode(output_prompt[i], skip_special_tokens=False)
            #     generated_output_text = pl_module.llm_tokenizer.decode(predicted_ids[i], skip_special_tokens=False)
            #     # print('target', target_text)
            #     # print('pred', generated_output_text)
            #     # print('----------------')
            #     f.write('target\n')
            #     f.write(target_text+'\n')
            #     f.write('predicted\n')
            #     f.write(generated_output_text+'\n')
            #     f.write('--------------------\n')
            #     extracted_target = pl_module.extract_prediction_values(target_text)
            #     extracted_pred = pl_module.extract_prediction_values(generated_output_text)
            #     # print("target",extracted_target)
            #     # print("pred",extracted_pred)
            # pl_module.train()

    # # @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # pass
        epoch_num = trainer.current_epoch
        # print(trainer.config.mode)
        save_name = "frame_1b_merge_30s_new_1st_stage"+str(epoch_num)+"_.ckpt"
        filename = os.path.join(self.save_dir, \
                save_name)
        self._save_checkpoint(trainer, pl_module, filename)

    # @rank_zero_only
    # def on_validation_end(self, trainer, pl_module):
    #     """
    #     Called when the validation ends.

    #     Args:
    #         trainer (Trainer): The PyTorch Lightning trainer.
    #         pl_module (LightningModule): The model being trained.
    #     """
    #     # Get the monitored metric
    #     metrics = trainer.logged_metrics
    #     epoch_num = trainer.current_epoch
    #     current_score = metrics.get(self.monitor).cpu().item()

    #     if f"epoch_{epoch_num}" not in self.all_epochs_metrics:
    #         self.all_epochs_metrics[f"epoch_{epoch_num}"]={}

    #     for key in metrics:
    #         self.all_epochs_metrics[f"epoch_{epoch_num}"][key]=metrics[key].cpu().item()

    #     # Initialize the best score if it's the first time
    #     if self.best_score is None:
    #         self.best_score = current_score
    #         self.all_epochs_metrics["best_score"]=self.best_score
    #         self.all_epochs_metrics["best_epoch"]=epoch_num

    #     # Save the checkpoint if the metric has improved
    #     is_improved = (
    #         (self.mode == "min" and current_score < self.best_score) or
    #         (self.mode == "max" and current_score > self.best_score)
    #     )
    #     previous_best_checkpoint = glob.glob(os.path.join(self.save_dir,"*best_checkpoint*"))

    #     if is_improved or len(previous_best_checkpoint)==0:
    #         self.best_score = current_score
    #         #delete previous checkpoint:
    #         if len(previous_best_checkpoint)>0:
    #             print("deleting previous checkpoint", previous_best_checkpoint[0])
    #             os.remove(previous_best_checkpoint[0])

    #         filename = os.path.join(self.save_dir, \
    #             self.filename_prefix+"_best_checkpoint_audio"+f"{epoch_num}"+".ckpt")
    #         self._save_checkpoint(trainer, pl_module, filename)
    #         self.all_epochs_metrics["best_score"]=self.best_score
    #         self.all_epochs_metrics["best_epoch"]=epoch_num
        
    #     # save metadata
    #     f=open(self.meta_data_path,"w")
    #     f.write(json.dumps(self.all_epochs_metrics, sort_keys=False, indent=2))
    #     f.close()

    def _save_checkpoint(self, trainer, pl_module, filename):
        """
        Save the model checkpoint.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (LightningModule): The model being trained.
        """
        
        trainer.save_checkpoint(filename)
        print(f"Checkpoint saved to: {filename}")

    # @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """
        Called at the end of training. Logs the best score.
        """
        print(f"Best {self.monitor}: {self.best_score}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass
        # This is the closest to `test_batch_end` in 
        # pl_module.eval()
        # # waveform, inputs, targets,instruction_len, data_ids,  = batch
        # waveform, pre_speech_prompt, post_speech_prompt, output_prompt, data_ids=batch
        
        # # print('hererere', instruction_len)
        
        # embeds, atts_mask, embeds_no_output, atts_mask_no_output, target_ids = pl_module.encode(waveform, pre_speech_prompt, post_speech_prompt, output_prompt,)
        # # print('here')
        # predicted_ids=pl_module.generate_text(embeds_no_output, atts_mask_no_output)
        # # print('heelo')
        # for i in range(len(predicted_ids)):   
        #     # input_ids_after = trainer.llm_tokenizer(message, add_special_tokens=False, return_tensors='pt').input_ids
        #     # target_ids_after = copy.deepcopy(input_ids_after[-1]) 
        #     target_text = pl_module.llm_tokenizer.decode(output_prompt[i], skip_special_tokens=False)
        #     generated_output_text = pl_module.llm_tokenizer.decode(predicted_ids[i], skip_special_tokens=False)
        #     # print('target', target_text)
        #     # print('pred', generated_output_text)
        #     # print('----------------')
        #     # f.write('{\n')
        #     # f.write('target\n')
        #     # f.write(target_text+'\n')
        #     # f.write('predicted\n')
        #     # f.write(generated_output_text+'\n')
        #     # f.write('}\n')
        #     # f.write('--------------------\n')
        #     extracted_target = pl_module.extract_prediction_values(target_text)
        #     extracted_pred = pl_module.extract_prediction_values(generated_output_text)
        #     # print("target",extracted_target)
        #     # print("pred",extracted_pred)
        # # print(f"Finished test batch {batch_idx}")

# Example of using the callback
# from pytorch_lightning import Trainer
# trainer = Trainer(callbacks=[MetricCheckpointCallback(monitor="val_loss", mode="min")])
