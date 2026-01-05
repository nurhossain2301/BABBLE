import torch
import torchaudio
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import lightning.pytorch as pl
import numpy as np
import random
import re
import json
import os
import copy
import ast
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, cohen_kappa_score, recall_score, precision_score
)
from difflib import get_close_matches
import subprocess
# import spyder
import pdb

from model.encoder import get_audio_encoder
from model.connector import get_connector
from model.linear import get_linear_layer
from model.llm import get_llm
from model.qformer import QFormer

class WhisperToWav2VecProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1280, 768)

    def forward(self, x):
        # x: [B, T, 1280] from Whisper encoder
        return self.proj(x)  # [B, T, 768]
    
class LBLLMLightningBase(pl.LightningModule):
    def __init__(self, 
                 audio_enc_dim=768,
                 speaker_enc_dim=768, 
                 llm_dim=2048, 
                 audio_encoder_name=None,
                 audio_model_path=None,
                 llm_model_path=None,
                 speaker_encoder_name=None,
                 speaker_encoder_path=None,
                 speech_enroll_prefix=None,
                 linear_neurons=1,
                 input_size=12,
                 connector_name='cnn',
                 llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 audio_length=2,
                 target_length=2,
                 finetune_encoder=False,
                 freeze_linear_layer=False,
                 connector_k=5,
                 use_lora=True,
                 lora_r=32,
                 lora_alpha=2,
                 max_lr=3e-4,
                 total_training_step=500000,
                 warmup_steps=1000,
                 apply_suggest_word=None,
                 prompt_format='json',
                 modify_time_constrain=True,
                 ref_rttm_prefix=None,
                 test_file_name=None,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_enc_dim = audio_enc_dim
        self.llm_dim = llm_dim
        self.llm_name = llm_name
        self.speaker_encoder_name = speaker_encoder_name
        self.finetune_encoder = finetune_encoder
        self.use_lora = use_lora

        # print(audio_model_path)
        self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder, model_path=audio_model_path, output_all_hiddens=False)
        self.weighted_average = get_linear_layer(linear_neurons, input_size, freeze=freeze_linear_layer)
        self.text_file = "whisper_1b_frame_10s_3rd_stage.txt"
        # self.text_file = "result_8b_caption.txt"
        self.f = open(self.text_file, 'w')
        
        ### set up speaker embedding network
        if speaker_encoder_name is not None:
            if not finetune_encoder and speaker_encoder_name=="wav2vec-LL4300": # share the same frozen audio encoder with speaker encoder
                self.spk_emb_encoder = self.audio_encoder 
            else:
                self.spk_emb_encoder = get_audio_encoder(speaker_encoder_name, finetune_encoder=False, model_path=speaker_encoder_path)

            self.weighted_average_spk=None
            if speaker_encoder_name=="wav2vec-LL4300": # share the same frozen audio encoder with speaker encoder
                self.weighted_average_spk = get_linear_layer(linear_neurons, input_size)
            self.dnn_block = get_linear_layer(llm_dim, speaker_enc_dim, name="dnn_block")
            self.spk_enroll_emb={}
            self.speech_enroll_prefix=speech_enroll_prefix 
        #####
        self.connector = get_connector(connector_name, audio_enc_dim, llm_dim, connector_k)

        # self.qformer = QFormer(llm_dim=self.llm_dim) ##make it dynamic
        self.llm_tokenizer, self.llm_model = get_llm(llm_name, use_lora, lora_r, lora_alpha, llm_model_path=llm_model_path)
        
        self.max_lr = max_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps

        self.apply_suggest_word=apply_suggest_word
        self.modify_time_constrain=modify_time_constrain
        self.audio_length=audio_length
        self.target_length=target_length

        self.spk_dict_map={"":0,"infant":1,"female":2,"male":3,"child":4,\
        "infant_female":5,"infant_male":6,"infant_child":7, "female_male":8,\
        "female_child":9,"male_child":10}
        self.irre_spk_dict_map={"":0,"irrelevant female":1,"irrelevant male":2,"irrelevant overlap":3}
        self.voc_dict_map={
            "infant":{"crying":1,"fussing":2,"babbling":3,"laughter":4},
            "female":{"child-directed speech":1,"adult-directed speech":2,"laughter":3,"singing":4},
            "male":{"child-directed speech":1,"adult-directed speech":2},
            "child":{"speech":1}
        }
    
        self.spk_code_map={"infant":"CHN","female":"FAN","male":"MAN","child":"CXN","spk":"spk"}
        self.voc_code_map={
            "infant":{"crying":"CRY","fussing":"FUS","babbling":"BAB","laughter":"LAU"},
            "female":{"child-directed speech":"CDS","adult-directed speech":"FAN","laughter":"LAU","singing":"SNG"},
            "male":{"child-directed speech":"CDS","adult-directed speech":"MAN"},
            "child":{"speech":"CXN"}
        }

        self.all_possible_voc=["infant crying", "infant fussing", "infant babbling", "infant laughter",\
                                "female child-directed speech", "female adult-directed speech", "female laughter", "female singing",\
                                "male child-directed speech", "male adult-directed speech", "child speech", \
                                "irrelevant female speech", "irrelevant male speech"]
        
  
        self.prompt_format=prompt_format
        self.ref_rttm_prefix=ref_rttm_prefix
        self.test_file_name=test_file_name
    
    def configure_optimizers(self):
        pass

    def forward(self, embeds, atts_mask, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts_mask,
            labels=label_ids,
        )
        return out

    def generate_text(self, inputs_embeds, attn_mask, max_new_tokens=256, temperature=0.2, top_p=0.9):
        with torch.no_grad():
            output = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                attention_mask=attn_mask,
                top_p=top_p,
                do_sample=True,  # Use sampling for more natural text
                # pad_token_id=self.llm_tokenizer.eos_token
            )
        return output

    def training_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_start(self):
        # initialize the entire array        
        self.clear_evaluation_array()

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx)

    # def on_validation_epoch_end(self):
    #     self.mode="test"
    #     print("Compute metrics for entire test set")
    #     self.evaluate_final_results()
    
    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx)

    def on_test_epoch_start(self):
        # initialize the entire array        
        self.clear_evaluation_array()

    def on_test_epoch_end(self):
        self.mode="test"
        print("Compute metrics for entire test set")
        self.evaluate_final_results()
        #self.output_rttm_file()
        #self.compute_der()

    def evaluation_step(self, batch, batch_idx):
        pass

    def compute_metrics(self, pred_dict, target_dict):
        pass

    def evaluate_final_results(self):
        # print(self.pred_dict)
        # print(self.target_dict)
        try:
            if torch.distributed.get_world_size()>1:
                for key in self.pred_dict:
                    pred_tensor = torch.tensor(self.pred_dict[key]) if not isinstance(self.pred_dict[key], torch.Tensor) else self.pred_dict[key]
                    target_tensor = torch.tensor(self.target_dict[key]) if not isinstance(self.target_dict[key], torch.Tensor) else self.target_dict[key]
                    self.final_pred_dict[key] = self.all_gather(pred_tensor)
                    self.final_target_dict[key] = self.all_gather(target_tensor)
                    
            if self.trainer.is_global_zero:
                if torch.distributed.get_world_size()==1:
                    self.final_pred_dict, self.final_target_dict=self.pred_dict, self.target_dict
                for key in self.final_pred_dict:
                    if isinstance(self.final_pred_dict[key], torch.Tensor):
                        self.final_pred_dict[key]=self.final_pred_dict[key].detach().cpu().numpy().flatten()
                        self.final_target_dict[key]=self.final_target_dict[key].detach().cpu().numpy().flatten()
                print(len(self.final_pred_dict[key]), len(self.final_target_dict[key]))
                self.compute_metrics(self.final_pred_dict, self.final_target_dict)
        except:
            self.final_pred_dict, self.final_target_dict=self.pred_dict, self.target_dict
            self.compute_metrics(self.final_pred_dict, self.final_target_dict)  

    def clear_evaluation_array(self):
        self.target_dict={"spk":[],"irrelevant_spk":[],"infant":[],"female":[],"male":[],"child":[]}
        self.pred_dict={"spk":[],"irrelevant_spk":[],"infant":[],"female":[],"male":[],"child":[]}
        self.output_rttm={"spk":[],"infant":[],"female":[],"male":[],"child":[]}
        self.final_pred_dict, self.final_target_dict={},{}

    def suggest_similar_words(self, predicted_word, target_list):
        out=get_close_matches(predicted_word, target_list, cutoff=0.6)
        return out[0] if len(out)>0 else ""

    def compute_metrics(self, pred_dict, target_dict):
        self.summary={}
        for key in pred_dict:
            pred_dict[key]=np.asarray(pred_dict[key])
            target_dict[key]=np.asarray(target_dict[key])
            #print(key, len(pred_dict[key]),len(target_dict[key]))
             
        pred_dict["spk_ovl"]=np.where(pred_dict["spk"]>=5, 5, pred_dict["spk"])
        target_dict["spk_ovl"]=np.where(target_dict["spk"]>=5, 5, target_dict["spk"])

        for curr_type in pred_dict:
            preds = pred_dict[curr_type]
            labels = target_dict[curr_type]
            self.summary[f"accuracy{curr_type}"]=round(accuracy_score(labels,preds),3)
            self.summary[f"macro_f1{curr_type}"]=round(f1_score(labels,preds,average="macro"),3)
            self.summary[f"kappa{curr_type}"]=round(cohen_kappa_score(labels,preds),3)
            self.summary[f"confusion_matrix{curr_type}"]=confusion_matrix(labels,preds)

        message = f"Epoch number: {self.current_epoch}\n"
        if self.apply_suggest_word:
            message += f"Applying suggested word with method of {self.apply_suggest_word}\n"
        if self.modify_time_constrain:
            message += f"Modify time constrain\n"
        if self.test_file_name:
            message += f"Evaluating test file: {self.test_file_name}\n"
        # print speaker diarization performance
        for tier in pred_dict:
            message += f"Accuracy: {self.summary[f'accuracy{tier}']}\n"
            message += f"macro f1 score: {self.summary[f'macro_f1{tier}']}\n"
            message += f"kappa: {self.summary[f'kappa{tier}']}\n"
            message += f"Confusion matrix:\n"
            if tier =="spk":
                class_names = ["SIL", "CHN", "FAN", "MAN", "CXN", "CHN_FAN", "CHN_MAN", "CHN_CXN", "FAN_MAN", "FAN_CXN", "MAN_CXN"]
            if tier =="spk_ovl":
                class_names = ["SIL", "CHN", "FAN", "MAN", "CXN", "OVL"]
            if tier =="irrelevant_spk":
                class_names = ["SIL", "SEC_FAN", "SEC_MAN", "OVL"]
            if tier =="infant":
                class_names = ["SIL", "CRY", "FUS", "BAB", "LAU"]
            if tier =="female":
                class_names = ["SIL", "CDS", "FAN", "LAU", "SNG"]
            if tier =="male":
                class_names = ["SIL", "CDS", "MAN"]
            if tier =="child":
                class_names = ["SIL", "SPEECH"]

            try:
                df_confusion_matrix = pd.DataFrame(self.summary[f'confusion_matrix{tier}'], index=class_names, columns=class_names)
            except:
                df_confusion_matrix = pd.DataFrame(self.summary[f'confusion_matrix{tier}'], index=None, columns=None)
            message += df_confusion_matrix.to_string()+"\n"

        overall_macro_f1 = 0.25*self.summary["macro_f1spk_ovl"] + 0.25 * self.summary["macro_f1irrelevant_spk"] + \
                        0.5*(0.2*self.summary["macro_f1infant"] + 0.2*self.summary["macro_f1female"]+0.1*self.summary["macro_f1male"])
        overall_macro_f1 = round(overall_macro_f1, 3)
        self.summary["overall_macro_f1"]=overall_macro_f1
        message += f"Overall macro f1 score: {overall_macro_f1}\n"
        message += "---------------------------------------------"

        print(message)
        if self.logger:
            if not os.path.exists(self.logger.log_dir):
                os.mkdir(self.logger.log_dir)
            f=open(os.path.join(self.logger.log_dir,f"detailed_output_{self.mode}.txt"),"a")
            f.write(message)
            f.close()
        
        self.log("val_f1_score", self.summary["overall_macro_f1"], rank_zero_only=True, on_epoch=True)
        self.log("val_f1_spk_ovl", self.summary["macro_f1spk_ovl"], rank_zero_only=True, on_epoch=True)
        self.log("val_f1_irrelevant_spk", self.summary["macro_f1irrelevant_spk"], rank_zero_only=True, on_epoch=True)
        self.log("val_f1_infant", self.summary["macro_f1infant"], rank_zero_only=True, on_epoch=True)
        self.log("val_f1_female", self.summary["macro_f1female"], rank_zero_only=True, on_epoch=True)
        self.log("val_f1_male", self.summary["macro_f1male"], rank_zero_only=True, on_epoch=True)

    def compute_der(self):
        output=""
        if self.test_file_name:
            output += f"Evaluating test file: {self.test_file_name}\n"
        for key in self.output_rttm:
            output += f"Evaluate {key} \n"
            ref_rttm_path = f"{self.ref_rttm_prefix}/test_{self.spk_code_map[key]}.rttm"
            hyp_rttm_path = f"{self.logger.log_dir}/rttm_{self.spk_code_map[key]}_test.rttm"

            command=["spyder",ref_rttm_path, hyp_rttm_path, "-c", "0.25"]
            output += subprocess.check_output(command, text=True)
        f=open(os.path.join(self.logger.log_dir,f"der_output_{self.mode}.txt"),"a")
        f.write(output)
        f.close()



class FrameTrainer(LBLLMLightningBase):
    """ 
    For frame-based LB-LLM training
    """   

    def encode(self, waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids):
        batch_size = waveform.shape[0]
        # print(self.audio_encoder(waveform).shape)
        
        speech_embeds = self.audio_encoder(waveform)
        # speech_embeds = self.weighted_average(speech_embeds) #when all layer is used
        speech_embeds = self.connector(speech_embeds)
        # speech_embeds = self.qformer(speech_embeds)
        
        if self.use_lora:
            embedder = self.llm_model.model.model.embed_tokens
        else:
            embedder = self.llm_model.model.embed_tokens

        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        combined_embeds = torch.cat([pre_prompt_embeds, speech_embeds, post_prompt_embeds, output_prompt_embeds], dim=1)
        atts_mask = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        input_token_length = pre_tokenized_ids.shape[1] + speech_embeds.shape[1] + post_tokenized_ids.shape[1]
        # input_token_length = pre_tokenized_ids.shape[1] + speech_embeds.shape[1] 
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)

        combined_embeds_no_output = torch.cat([pre_prompt_embeds, speech_embeds, post_prompt_embeds], dim=1)
        atts_mask_no_output = torch.ones(combined_embeds_no_output.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        # combined_embeds_no_output = torch.cat([pre_prompt_embeds, speech_embeds], dim=1)
        # atts_mask_no_output = torch.ones(combined_embeds_no_output.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        return combined_embeds, atts_mask, combined_embeds_no_output, atts_mask_no_output, label_ids

    def configure_optimizers(self):
        opt = [
            # {"params": self.audio_encoder1.parameters(), "lr": 2e-5},
            {"params": self.audio_encoder.parameters(), "lr": 2e-5},
            # {"params": self.weighted_average.parameters(), "lr": self.max_lr},
            {"params": self.connector.parameters(), "lr": self.max_lr},
            # {"params": self.qformer.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": 1e-5},
        ]
        optimizer = AdamW(opt, lr=self.max_lr, betas=(0.9, 0.95), weight_decay=0.01)
        return optimizer
     
    def training_step(self, batch, batch_idx):
        waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, _ = batch
        # waveform, pre_tokenized_ids, output_tokenized_ids, _ = batch
        embeds, atts_mask, _, _, label_ids = self.encode(waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.forward(embeds, atts_mask, label_ids)
        #apply constraint decoding and apply categorical loss and 
        loss =  outputs["loss"]
        self.log("train/loss", loss, on_epoch=False)
        return loss

    def evaluation_step(self, batch, batch_idx):
        waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, data_ids = batch
        # waveform, pre_tokenized_ids, output_tokenized_ids, _ = batch
        embeds, atts_mask, embeds_no_output, atts_mask_no_output, label_ids = self.encode(waveform, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.forward(embeds, atts_mask, label_ids)
        loss = outputs["loss"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        predicted_ids=self.generate_text(embeds_no_output, atts_mask_no_output)
        # f = open('record_test_batch_main_test.txt', 'w')
        for i in range(len(predicted_ids)):    
            target_text = self.llm_tokenizer.decode(output_tokenized_ids[i], skip_special_tokens=False)
            generated_output_text = self.llm_tokenizer.decode(predicted_ids[i], skip_special_tokens=False)
            self.f.write('target: '+ target_text+'\n')
            self.f.write('pred: '+ generated_output_text+'\n')
            extracted_target = self.extract_prediction_values(target_text)
            extracted_pred = self.extract_prediction_values(generated_output_text)
            # print("target",extracted_target)
            # print("pred",extracted_pred)

            target_dict = self.convert_output(extracted_target, ground_truth=True, data_ids = data_ids)
            pred_dict = self.convert_output(extracted_pred, ground_truth=False, data_ids = data_ids)
            

            for key in self.target_dict:
                self.target_dict[key].extend(target_dict[key])
                self.pred_dict[key].extend(pred_dict[key])
        return {"val_loss": loss}        
 
    def extracted_output_array(self, input_string):
        segments = input_string.split(",")[:int(self.target_length*10)] 
        pattern = r'(\|.*?\|)(.*?)(\|.*?\|)'
        out_array=[""]*int(self.target_length*10)
        for i,segment in enumerate(segments):
            try:
                spk_vocs = segment.split(";")
                for spk_voc in spk_vocs:
                    values = list(re.findall(r'(\|.*?\|)(.*?)(\|.*?\|)', spk_voc)[0])
                    for item in values:
                        if item.startswith("|"):
                            out_array[i]+=item
                    out_array[i]+=";"
                if out_array[i].endswith(";"):
                    out_array[i]=out_array[i][:-1]
            except:
                continue
        return out_array

    
    def extract_prediction_values(self, input_string):
        items = re.findall(r"\[.*?\]", input_string)
        # print('tems:', items)
        try:
            str_match = re.search(r"\[([^\[\]]+)\]", items[0])
        except:
            # print('here')
            str_match=['']
        
        # str_match = match.group(1)
        try:
            json_str = [str_match.group(1)]
        except:
            json_str = ['']
        # print(json_str)
        return json_str
    
    def convert_output(self, input_array, ground_truth=False, data_ids = None):
            
        # target_length=0.1
        # apply_suggest_word = 'combined'
        # output_array_length = int(target_length*10)
        output_array_length = int(self.target_length*10)
        spk_out_dict={"infant":np.zeros(output_array_length),"female":np.zeros(output_array_length),\
                    "male":np.zeros(output_array_length),"child":np.zeros(output_array_length),\
                    "irrelevant male":np.zeros(output_array_length),"irrelevant female":np.zeros(output_array_length)}
        voc_out_dict={"infant":{"crying":np.zeros(output_array_length), "fussing":np.zeros(output_array_length), \
                    "babbling":np.zeros(output_array_length), "laughter":np.zeros(output_array_length)},
                "female":{"child-directed speech":np.zeros(output_array_length), "adult-directed speech":np.zeros(output_array_length), \
                    "laughter":np.zeros(output_array_length), "singing":np.zeros(output_array_length)},
                "male":{"child-directed speech":np.zeros(output_array_length), "adult-directed speech":np.zeros(output_array_length)},
                "child":{"speech":np.zeros(output_array_length)},
        }
        for i,spk_vocs in enumerate(input_array):
            spk_vocs = spk_vocs.split(";")
            spk, voc = None, None
            for spk_voc in spk_vocs:
                matches = spk_voc.split('-')
                if len(matches) == 2:
                    spk, voc = matches[0], matches[1]
                if not spk or not voc:
                    continue

                if self.apply_suggest_word=="combined": 
                    if (spk not in spk_out_dict) or (spk in voc_out_dict and voc not in voc_out_dict[spk]):
                        #print("before", spk, voc)
                        curr_combine_word=spk+" "+voc
                        combined_words = self.suggest_similar_words(curr_combine_word, self.all_possible_voc)
                        # print("combined words", combined_words)
                        if combined_words!="":
                            combined_words = combined_words.split()
                            if combined_words[0]=="irrelevant":
                                spk = combined_words[0]+" "+combined_words[1] # irrelevant speaker
                                voc = " ".join(combined_words[2:])
                            else:
                                spk = combined_words[0]
                                voc = " ".join(combined_words[1:])                                
                            #print("after", spk, voc)

                    if spk in spk_out_dict:
                        spk_out_dict[spk][i]=1

                    if spk in voc_out_dict and voc in voc_out_dict[spk]:
                        voc_out_dict[spk][voc][i]=1
                else: # greedy decoding
                    if spk in spk_out_dict:
                        spk_out_dict[spk][i]=1
                    if spk in voc_out_dict and voc in voc_out_dict[spk]:
                        voc_out_dict[spk][voc][i]=1

        out_dict={"spk":[],"irrelevant_spk":[],"infant":[],"female":[],"male":[],"child":[]}
        
        # convert raw output into final output
        for i in range(output_array_length):
            curr_spk=""
            for spk in ["infant","female","male","child"]: # primary speakers
                if spk_out_dict[spk][i]==1:
                    curr_spk+=spk+"_"

            curr_spk=curr_spk[:-1]
            if curr_spk in self.spk_dict_map: # less than three speakers
                out_dict["spk"].append(self.spk_dict_map[curr_spk])
            else:
                out_dict["spk"].append(0)

            if spk_out_dict["irrelevant female"][i]==1 and spk_out_dict["irrelevant male"][i]==1: # secondary speakers
                out_dict["irrelevant_spk"].append(self.irre_spk_dict_map["irrelevant overlap"])
            elif spk_out_dict["irrelevant female"][i]==1:
                out_dict["irrelevant_spk"].append(self.irre_spk_dict_map["irrelevant female"])
            elif spk_out_dict["irrelevant male"][i]==1:
                out_dict["irrelevant_spk"].append(self.irre_spk_dict_map["irrelevant male"])
            else:
                out_dict["irrelevant_spk"].append(self.irre_spk_dict_map[""])

            for spk in ["infant","female","male","child"]:
                find_voc=False
                for voc,voc_idx in self.voc_dict_map[spk].items():
                    if voc_out_dict[spk][voc][i]==1:
                        out_dict[spk].append(voc_idx)          
                        find_voc=True
                        break
                if not find_voc:
                    out_dict[spk].append(0)
        for key in out_dict:
            out_dict[key]=np.asarray(out_dict[key])

        return out_dict
    
    def output_rttm_file(self):
        if self.logger:
            if not os.path.exists(self.logger.log_dir):
                os.mkdir(self.logger.log_dir)
        
        for key in self.output_rttm:
            f=open(os.path.join(self.logger.log_dir,f"rttm_{self.spk_code_map[key]}_{self.mode}.rttm"),"a")
            for item in self.output_rttm[key]:
                curr_id = "_".join(item[0].split("_")[:-2])
                label = item[1]
                duration = round(item[3]-item[2],1)
                start_time = round(float(item[2]),1)
                f.write(f"SPEAKER {curr_id} 1 {start_time:.2f} {duration:.2f} <NA> <NA> {label} <NA> <NA>\n")
            f.close()
    
    