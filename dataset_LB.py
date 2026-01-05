import torch
from transformers import AutoProcessor, AutoFeatureExtractor
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import torchaudio
import pandas as pd
import random
import numpy as np
import json
import os
import copy
from torch.nn.functional import one_hot

IGNORE_INDEX = -100

class MyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        

    def __call__(self, batch):
        # Separate the batch into components
        waveforms = [item[0].unsqueeze(0) for item in batch]
        pre_speech_prompts = [item[1] for item in batch]
        post_speech_prompts = [item[2] for item in batch]
        output_prompts = [item[3] for item in batch]
        data_ids = [item[5] for item in batch]

        output_tokenized_ids = self.tokenizer(
            [self.tokenizer.bos_token + prompt + self.tokenizer.eos_token for prompt in output_prompts],
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]
        
        
        # Tokenize prompts
        pre_tokenized_ids = self.tokenizer(
            pre_speech_prompts, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]
        
        post_tokenized_ids = self.tokenizer(
            post_speech_prompts, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]

        
        
        # Stack waveforms into a batch
        waveforms = torch.cat(waveforms, dim=0)


        # return waveforms, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, data_ids
        return waveforms, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, data_ids

class MyCollatorAudio:
    def __init__(self):
        # self.tokenizer = tokenizer
        self.num_classes = 5
        self.label2id = {"[No_sound]": 0, "[infant-babbling]": 1, "[infant-crying]": 2, "infant-fussing": 3, "[infant-laughter]": 4, "[female-child_directed_speech]": 5, "[female-laughter]": 6, "[female-singing]": 7, "[female-adult_directed_speech]": 8, \
                        "[male-adult_directed_speech]": 9, "[male-child_directed_speech]": 10, "[male-laughter]": 12, "[irrelevant_female-child_directed_speech]": 5, \
                        "[irrelevant_female-speech]": 8, "[irrelevant_male-speech]": 9, "[child-speech]": 11, "[child-laughter]": 11, "[child-child_directed_speech]": 11, "[child-singing]": 11, \
                        "[irrelevant_male-child_directed_speech]": 10, "[irrelevant_male-adult_directed_speech]": 9, "[irrelevant_female-adult_directed_speech]": 8, "[male-singing]": 12}
        self.CHN = {"[No_sound]": 0, "[infant-babbling]": 1, "[infant-crying]": 2, "[infant-fussing]": 3, "[infant-laughter]": 4, "[female-child_directed_speech]": 0, "[female-laughter]": 0, "[female-singing]": 0, "[female-adult_directed_speech]": 0, \
                        "[male-adult_directed_speech]": 0, "[male-child_directed_speech]": 0, "[male-laughter]": 0, "[irrelevant_female-child_directed_speech]": 0, \
                        "[irrelevant_female-speech]": 0, "[irrelevant_male-speech]": 0, "[child-speech]": 0, "[child-laughter]": 11, "[child-child_directed_speech]": 0, "[child-singing]": 0, \
                        "[irrelevant_male-child_directed_speech]": 0, "[irrelevant_male-adult_directed_speech]": 0, "[irrelevant_female-adult_directed_speech]": 0, "[male-singing]": 0, "[irrelevant_female-laughter]": 0, "[irrelevant_female-singing]": 0}

    def __call__(self, batch):
        # Separate the batch into components
        waveforms = [item[0].unsqueeze(0) for item in batch]
        pre_speech_prompts = [item[1] for item in batch]
        post_speech_prompts = [item[2] for item in batch]
        output_prompts = [item[3] for item in batch]
        data_ids = [item[5] for item in batch]

        new_prompts = []
        for prompt in output_prompts:
            prompt = prompt.split(';')
            if len(prompt)>1:
                # print(prompt, len(prompt))
                new_prompts.append(prompt[0]+']')
                # new_prompts.append('[' + prompt[1])
            else:
                new_prompts.append(prompt[0])
        labels = [self.CHN[l] for l in new_prompts]
        # print(labels)
        labels = torch.tensor(labels, dtype=torch.long)

        # One-hot encode
        labels_onehot = one_hot(labels, num_classes=self.num_classes).float()
        
        # Tokenize prompts
        
        
        
        # Stack waveforms into a batch
        waveforms = torch.cat(waveforms, dim=0)
        # print(waveforms.shape)


        # return waveforms, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, data_ids
        return waveforms, labels_onehot

class LBAudioDatasetCaption:
    def __init__(self, json_file, apply_1_shot=False, include_voc_count=False, mode='train', prompt_format='json'):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        f = open(json_file,"r")
        self.data_json = json.load(f)
        self.data_keys = list(self.data_json.keys())
        self.spk_dict={"CHN":"infant","FAN":"female","MAN":"male","CXN":"child","FAN2":"irrelevant female","MAN2":"irrelevant male"}
        self.voc_dict={"FUS":"fussing","CRY":"crying","BAB":"babbling",
                "CDS":"child-directed speech", "PLA":"child-directed speech", "PLAC":"child-directed speech",\
                "FAN":"adult-directed speech", "MAN":"adult-directed speech", "LAU":"laughter", "LAUC":"laughter",\
                "SNG":"singing", "SNGC":"singing"}
        
        
    
        self.apply_1_shot = apply_1_shot
        self.include_voc_count = include_voc_count
        self.prompt_format=prompt_format



    def __len__(self):
        return len(self.data_keys)

    def read_audio(self, waveforms_obj):
        if isinstance(waveforms_obj, str):
            audio, _ = torchaudio.load(waveforms_obj)
            return audio.transpose(0, 1).squeeze(1)

        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # Default stop to start -> if not specified, num_frames becomes 0,
        # which is the torchaudio default
        stop = waveforms_obj.get("stop", start)
        num_frames = stop - start
        audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
        if audio.shape[1] != 32001:
            audio = audio[:, :32000]
        audio = audio.transpose(0, 1)
        return audio.squeeze(1)

    def __getitem__(self, idx):
        #print(self.data_keys[idx])
        entry = self.data_json[self.data_keys[idx]]
        start_time = float(self.data_keys[idx].split("_")[-2])
        waveform = self.read_audio(entry["wav"])
        instruction_phrase = entry['question']

        pre_speech_prompt = f"Question:\n{instruction_phrase}"

        pre_speech_prompt += "\n\nInput:\n<speech>"
        post_speech_prompt = f"</speech>\n\n" + \
             "Output:\n"
        

        output_prompt = entry["caption"]

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]


class LBAudioDataset:
    def __init__(self, json_file, apply_1_shot=False, include_voc_count=False, mode='train', prompt_format='json'):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        f = open(json_file,"r")
        self.data_json = json.load(f)
        self.data_keys = list(self.data_json.keys())
        self.spk_dict={"CHN":"infant","FAN":"female","MAN":"male","CXN":"child","FAN2":"irrelevant female","MAN2":"irrelevant male"}
        self.voc_dict={"FUS":"fussing","CRY":"crying","BAB":"babbling",
                "CDS":"child-directed speech", "PLA":"child-directed speech", "PLAC":"child-directed speech",\
                "FAN":"adult-directed speech", "MAN":"adult-directed speech", "LAU":"laughter", "LAUC":"laughter",\
                "SNG":"singing", "SNGC":"singing"}
        
        self.instruction_phrases_train_with_voc_count = [
            "First count the number of vocalized segment in the audio, then detect the starting and ending timestamps of speaker and vocalization type for each segment.",
            "First, figure out how many vocal segments there are in the audio, then find the start and end times for each speaker and vocal type.",
            "Start by counting the vocalized sections in the audio, then note down the timestamps of the speaker and their vocalization type.",
            "Count the number of vocal segments first, then determine the starting and ending points of each speaker's vocalizations.",
            "Identify how many voiced parts are in the audio, then detect when each starts and ends along with the speaker and type.",
            "Locate the vocalized portions in the audio and record their beginning and ending times, as well as the speaker and type.",
            "First, check the total number of vocalized segments, then track the timestamps for the speaker and vocalization type of each.",
            "Start by finding the voiced sections in the audio, then identify the times they begin and end, along with speaker details.",
            "Determine the count of vocalized parts, then capture the timestamps for the speaker and the type of vocalization.",
            "Find out how many vocal segments are present, then mark the start and stop times with details on the speaker and type.",
            "Count the voiced sections, then note down when they begin and finish, specifying the speaker and vocal type.",
            "First, tally up the number of vocalized portions, then pinpoint their start and stop times along with speaker information.",
            "Begin by identifying all the voiced parts in the audio, then determine their timestamps and who is speaking.",
            "Figure out the total number of vocal segments and detect the starting and ending points for each speaker and type.",
            "Check how many voiced parts exist in the audio and record when they start and end, including the speaker and type.",
            "Start by assessing the vocalized sections in the audio, then capture the beginning and ending times along with speaker info.",
            "Identify the number of voiced portions, then track the timestamps and include the details of each speaker and vocalization type.",
            "Locate all vocalized parts in the audio and determine when they begin and finish, as well as their speaker and type.",
            "Find the vocal segments first, then detect the timestamps for when each starts and ends, along with speaker information.",
            "First, count the voiced portions of the audio, then note down the times they occur and the speaker and type of vocalization.",
            "Check the audio for vocalized segments, then figure out their starting and stopping points, and identify the speaker and type.",
            "Count the total vocalized parts, then mark their timestamps and include details about the speaker and vocal type.",
            "First, locate the vocalized sections, then determine the start and stop times for the speaker and their vocalization type.",
            "Begin by counting the voiced portions of the audio, then capture their timestamps and the speaker information.",
            "Count how many vocal segments are in the audio, then detect when they begin and end, and who is speaking.",
            "Start by identifying the number of vocalized parts, then record the timestamps along with the speaker and vocal type.",
            "Find out the number of voiced parts, then pinpoint their starting and ending times along with the speaker details.",
            "Locate the vocalized portions in the audio and determine their timestamps and the speaker and type of vocalization.",
            "Figure out how many vocal parts there are, then note when they start and stop, and include speaker information.",
            "First, identify the voiced sections of the audio, then track their timestamps and record the speaker and vocal type.",
            "Check how many vocalized parts exist, then find out when they occur and who is speaking, along with the vocalization type.",
        ]
        self.instruction_phrases_valid_with_voc_count = [
            "Count how many vocal segments there are, then pinpoint when they start and stop and note the speaker and type of vocalization.",
            "Begin by checking the number of vocalized parts, then track their timestamps and details about the speaker and type.",
            "Determine the total number of voiced sections, then mark when they begin and finish and include the speaker and type.",
            "First, find out how many vocal segments exist, then record the times they occur along with speaker and vocal type info.",
            "Locate the vocalized sections in the audio, then note when they start and end, and identify the speaker and vocalization type.",
            "Find all the voiced portions, then determine their timestamps and record the details of the speaker and type.",
            "Check the audio for vocalized segments, then capture the start and stop points along with the speaker and vocal type.",
            "First, identify the number of vocalized parts, then figure out their timestamps and include the speaker and type of vocalization.",
            "Count the vocal segments and detect when each begins and ends, including speaker details and type of vocalization.",
            "Start by locating the vocalized sections, then track their beginning and end points along with the speaker and vocalization type."
        ]
        #optional 10 test with voc count
        # self.instruction_phrases_test_with_voc_count = [
        #     "Determine the total vocalized portions, then mark their timestamps along with the speaker and type details.",
        #     "First, locate the voiced sections in the audio, then capture their start and stop times along with the speaker and type.",
        #     "Find all the vocalized parts, then figure out their beginning and ending points and specify the speaker and vocal type.",
        #     "Start by checking how many vocal parts are present in the audio, then note the timestamps and the speaker details.",
        #     "Count the voiced portions in the audio and record their start and end times along with the speaker and vocalization type.",
        #     "Begin by counting the vocalized segments, then capture the timestamps and identify the speaker and type for each.",
        #     "First, figure out how many vocal parts there are, then detect when they begin and stop and include speaker details.",
        #     "Find out the number of voiced sections, then mark their start and stop times with information on the speaker and type.",
        #     "Start by locating the vocalized portions, then record the timestamps along with the details of the speaker and vocal type.",
        #     "Identify the voiced parts in the audio, then determine their timestamps and specify the speaker and vocalization type.",
        # ]
        # use only 1 instruction prompt
        self.instruction_phrases_test_with_voc_count = [
            "Find out the number of voiced sections, then mark their start and stop times with information on the speaker and type.",
        ]

        self.instruction_phrases_set_train = [
            "Detect the starting and ending timestamps of speaker and vocalization type for each vocalized segment in the audio.",
            "Find the start and end times of each vocal segment, noting the speaker and vocalization type.",
            "Determine when each vocalized segment begins and ends, including speaker and vocalization details.",
            "Identify the timestamps marking the beginning and end of each speaker's vocalization type.",
            "Pinpoint the start and stop times for speakers and vocalization types in the audio segments.",
            "Track the timestamps for when each vocalization occurs and who the speaker is.",
            "Locate the start and end times of every vocalized segment, specifying the speaker and type.",
            "Analyze each vocalized part to record its timestamps and identify the speaker and vocalization type.",
            "Detect the beginning and ending times of every vocalized segment, along with speaker details.",
            "Mark the timestamps for the start and end of each vocalization type and speaker.",
            "Record the time intervals for all vocalized segments, identifying the speaker and type of vocalization.",
            "Log when each speaker begins and ends their vocalization, specifying the type of sound.",
            "Trace the start and end points of each vocal segment with speaker and vocalization information.",
            "Specify the timestamps for the start and stop of vocalized segments and label the speaker.",
            "Identify the timeframes for each speaker's vocalizations and classify the type of sound.",
            "Locate the start and finish times of every vocalized segment with speaker and vocalization info.",
            "Find when each vocalized part starts and ends, and note the speaker and type of vocalization.",
            "Highlight the timestamps where vocalized segments occur, including speaker and sound type.",
            "Log the beginning and ending times of each vocal segment and identify the speaker and type.",
            "Determine the time intervals of vocalized segments, along with speaker and vocalization type.",
            "Capture the timestamps for each vocalized segment and classify the speaker and vocalization type.",
            "Record when each vocalized portion begins and ends, identifying the speaker and type of sound.",
            "Pinpoint the exact times for the start and end of vocalized segments with speaker details.",
            "Mark the intervals for each vocalized segment and label the speaker and type of vocalization.",
            "Track the start and stop times for every vocalized segment with speaker and sound type details.",
            "Determine the start and finish of each vocalization and note the associated speaker.",
            "Analyze the timestamps of vocalized segments and specify the speaker and vocalization type.",
            "Find and record the time intervals for each vocalized segment, including the speaker and type.",
            "Detect when vocalized segments start and stop, noting the speaker and vocalization type.",
            "Trace the beginning and end times of every vocalized portion and classify the speaker.",
        ]

        self.instruction_phrases_set_valid=[
            "Locate the timeframes for each vocalization and identify the associated speaker and type.",
            "Log the time intervals for vocalized segments, along with speaker and vocalization information.",
            "Identify the start and end points of each vocalized part, specifying the speaker and type.",
            "Capture the timestamps of vocalized segments and label the speaker and vocalization type.",
            "Determine the timing for each vocalized part and record the speaker and sound type.",
            "Highlight when vocalized segments begin and end, including speaker and type of sound.",
            "Pinpoint the exact start and stop times of vocalized segments with speaker details.",
            "Analyze the intervals of each vocalized segment and identify the speaker and vocalization type.",
            "Find the timestamps for vocalized parts and note the speaker and type of vocalization.",
            "Mark the start and end times for every vocalized segment, identifying the speaker and type.",
        ]

        # self.instruction_phrases_set_test=[
        #     "Record the timing for each vocalized part and classify the speaker and type of sound.",
        #     "Detect the start and stop times of vocalized segments and log the speaker and vocalization details.",
        #     "Trace the beginning and ending of each vocalized segment and specify the speaker and type.",
        #     "Identify the intervals where vocalizations occur and classify the speaker and sound type.",
        #     "Track the start and end points of each vocalized segment with speaker and vocalization labels.",
        #     "Find the time ranges for vocalized segments and note the speaker and type of sound.",
        #     "Log the timestamps for vocalized portions and identify the speaker and type of vocalization.",
        #     "Analyze when each vocalized segment starts and ends, including speaker and sound type.",
        #     "Pinpoint the exact times for vocalized segments and label the speaker and vocalization type.",
        #     "Capture the intervals for vocalized parts and specify the speaker and type of sound."  
        # ]

        # use only 1 instruction prompt
        self.instruction_phrases_set_test=[
            "Detect the start and stop times of vocalized segments and log the speaker and vocalization details.",
        ]

        self.instruction_phrases_set_with_voc_count={"train": self.instruction_phrases_train_with_voc_count, \
                                      "valid": self.instruction_phrases_valid_with_voc_count,\
                                      "test": self.instruction_phrases_test_with_voc_count
                                      }
        self.instruction_phrases_set={"train": self.instruction_phrases_set_train, \
                                      "valid": self.instruction_phrases_set_valid,\
                                      "test": self.instruction_phrases_set_test
                                      }
        self.apply_1_shot = apply_1_shot
        self.include_voc_count = include_voc_count
        self.prompt_format=prompt_format

        if self.include_voc_count:
            self.instruction_phrases=self.instruction_phrases_set_with_voc_count[mode]
        else:
            self.instruction_phrases=self.instruction_phrases_set[mode]


    def __len__(self):
        return len(self.data_keys)

    def read_audio(self, waveforms_obj):
        if isinstance(waveforms_obj, str):
            audio, _ = torchaudio.load(waveforms_obj)
            return audio.transpose(0, 1).squeeze(1)

        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # Default stop to start -> if not specified, num_frames becomes 0,
        # which is the torchaudio default
        stop = waveforms_obj.get("stop", start)
        num_frames = stop - start
        audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
        if audio.shape[1] == 32001:
            audio = audio[:, :32000]
        audio = audio.transpose(0, 1)
        return audio.squeeze(1)

    def __getitem__(self, idx):
        #print(self.data_keys[idx])
        entry = self.data_json[self.data_keys[idx]]
        start_time = float(self.data_keys[idx].split("_")[-2])
        waveform = self.read_audio(entry["wav"])
        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase}"
        if self.apply_1_shot:
            if self.include_voc_count:
                if self.prompt_format=="json":
                    pre_speech_prompt += '''Use the following json format for the output: e.g., {"number of vocalization": 2, "|infant| |babbling|": [0.3, 1.0], "|female| |child-directed speech|":[2.0,3.5]}.'''
                else: # plain text
                    pre_speech_prompt += '''Use the following format for the output: e.g., "number of vocalization heard is 2 / |infant| |babbling| is heard between 0.3 and 1.0 seconds / |female| |child-directed speech| is heard between 2.0 and 3.5 seconds.'''
            else:
                if self.prompt_format=="json":
                    pre_speech_prompt += '''Use the following json format for the output: e.g., {"|infant| |babbling|": [0.3, 1.0], "|female| |child-directed speech|":[2.0,3.5]}.'''
                else: # plain text
                    pre_speech_prompt += '''Use the following format for the output: e.g., "|infant| |babbling| is heard between 0.3 and 1.0 seconds / |female| |child-directed speech| is heard between 2.0 and 3.5 seconds.'''

        pre_speech_prompt += "\n\nInput:\n<speech>"
        post_speech_prompt = f"</speech>\n\n" + \
             "Output:\n"
        out_dict = {}
        output_prompt = "{"

        curr_count = 0
        for key, value in entry["label"].items():
            #print(key,value)
            spk = self.spk_dict[key.split("_")[0]]
            voc = key.split("_")[1]
            if spk.startswith("irrelevant") or spk=="child":
                voc="speech"
            elif voc in self.voc_dict:
                voc=self.voc_dict[voc]
            else:
                continue
            for start,end in value:
                if "|"+spk+"|"+" "+"|"+voc+"|" not in out_dict:
                    out_dict["|"+spk+"|"+" "+"|"+voc+"|"]=[]
                out_dict["|"+spk+"|"+" "+"|"+voc+"|"].append((round(start-start_time,1),round(end-start_time,1)))
                curr_count+=1

        if self.include_voc_count:
            if self.prompt_format=="json":
                output_prompt +=f'  "number of vocalization": {curr_count}, '
            else:
                output_prompt +=f' number of vocalization heard is {curr_count} /'

        for key,value in out_dict.items():
            for start,end in value:
                if self.prompt_format=="json":
                    output_prompt +=f'  "{key}": [{start},{end}], '
                else:
                    output_prompt +=f'  {key} is heard between {start} and {end} seconds /'

        output_prompt = output_prompt.rstrip(',\n').rstrip("/") + "}"

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]

class LBAudioDatasetFrameBased:
    def __init__(self, json_file, apply_1_shot=False, mode='train', audio_length=5, target_length=1):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        f = open(json_file,"r")
        self.data_json = json.load(f)
        self.data_keys = list(self.data_json.keys())
        self.spk_dict={"CHN":"infant","FAN":"female","MAN":"male","CXN":"child","FAN2":"irrelevant female","MAN2":"irrelevant male"}
        self.voc_dict={"FUS":"fussing","CRY":"crying","BAB":"babbling",
                "CDS":"child-directed speech", "PLA":"child-directed speech", "PLAC":"child-directed speech",\
                "FAN":"adult-directed speech", "MAN":"adult-directed speech", "LAU":"laughter", "LAUC":"laughter",\
                "SNG":"singing", "SNGC":"singing"}
        
        self.instruction_phrases_set_train = [
            "Identify the speaker and the types of vocalizations at each frame from the audio input.",
            "Determine who is speaking and the vocalization types at the frame level based on the audio.",
            "Listen to the audio and recognize both the speaker and the vocalization types in each frame.",
            "From the audio input, find out who is speaking and what types of vocalizations are present at each frame.",
            "Analyze the audio to detect both the speaker and the types of vocal sounds in each frame.",
            "Figure out the speaker and the vocalization types at the frame level from the audio.",
            "At the frame level, recognize the speaker and what kind of vocalizations are being made in the audio.",
            "Detect who is speaking and the types of vocal sounds in each frame of the audio.",
            "Identify the speaker and classify the vocalizations at each frame level using the audio input.",
            "Based on the audio, detect the speaker and the vocalization types at the frame level.",
            "In the given audio, identify both the speaker and the vocalization types per frame.",
            "Listen to the input and detect both the speaker and the vocalization type at each frame.",
            "Find out the speaker and the type of vocalization in each audio frame.",
            "At the frame level, recognize who is speaking and the vocalization types from the audio input.",
            "Analyze the audio at the frame level to figure out who is speaking and what type of vocalization it is.",
            "Use the audio input to detect the speaker and the types of vocalizations at the frame level.",
            "From the audio data, identify both the speaker and the vocalization types for each frame.",
            "Check the audio to recognize the speaker and classify the vocalization types frame by frame.",
            "Find the speaker and classify vocalizations at each frame in the audio input.",
            "Listen to the audio and identify both the speaker and the vocalization types in each frame.",
            "Using the audio, figure out the speaker and what types of vocal sounds are present per frame.",
            "At every frame, identify the speaker and the type of vocalization based on the audio input.",
            "Process the audio to detect the speaker and vocalization types frame by frame.",
            "Break down the audio and identify both the speaker and the vocalization types per frame.",
            "In each frame of the audio, determine the speaker and what types of vocalizations are used.",
            "Identify the speaker and vocalization types at the frame level by analyzing the audio.",
            "From the audio, detect both the speaker and the vocalization types at each frame.",
            "Using the audio input, recognize who’s speaking and what kind of vocalization appears per frame.",
            "Examine the audio input to identify the speaker and vocalizations for each frame.",
            "In each frame, detect both the speaker and the vocalization type from the audio.",
        ]

        self.instruction_phrases_set_valid = [
            "At every frame, identify who is speaking and what types of vocalizations are made in the audio.",
            "Analyze the audio to recognize the speaker and the types of vocalization in each frame.",
            "From the audio input, figure out both the speaker and the vocalization types frame by frame.",
            "Identify both the speaker and the vocalization types in the audio, one frame at a time.",
            "Look at the audio input and detect who is speaking and the types of vocalizations at each frame.",
            "Determine the speaker and the type of vocalization in each frame by listening to the audio.",
            "In each frame, figure out the speaker and classify the vocalization types in the audio.",
            "At the frame level, detect the speaker and the vocalization type from the audio input.",
            "Analyze the audio to detect the speaker and classify vocalizations at each frame.",
            "Listen to the audio and identify the speaker and vocalization types for each frame.",
        ]
        
        # optionlly 10 instructions for test set
        # self.instruction_phrases_set_test = [
        #     "Using the audio, identify both the speaker and the type of vocalization in each frame.",
        #     "At each frame, recognize the speaker and the types of vocal sounds in the audio.",
        #     "Recognize who’s speaking and what vocalization types are used at the frame level from the audio.",
        #     "Detect both the speaker and the types of vocalizations in each frame by examining the audio.",
        #     "In each frame of the audio, detect who’s speaking and what vocalization types are used.",
        #     "Examine the audio and identify the speaker and vocalization types frame by frame.",
        #     "From the audio input, classify the speaker and the vocalization types at the frame level.",
        #     "Recognize the speaker and detect the vocalization types at each frame from the audio input.",
        #     "Use the audio input to identify both the speaker and vocalization types in each frame."
        #     "At the frame level, detect the speaker and categorize the types of vocalization in the audio.",
        # ]

        self.instruction_phrases_set_test = [
            "At each frame, recognize the speaker and the types of vocal sounds in the audio.",
        ]

        self.instruction_phrases_set={"train": self.instruction_phrases_set_train, \
                                      "valid": self.instruction_phrases_set_valid,\
                                      "test": self.instruction_phrases_set_test
                                      }
        self.apply_1_shot = apply_1_shot
        self.instruction_phrases=self.instruction_phrases_set[mode]
        self.audio_length=audio_length
        self.target_length=target_length

    def __len__(self):
        return len(self.data_keys)

    def read_audio(self, waveforms_obj):
        if isinstance(waveforms_obj, str):
            audio, _ = torchaudio.load(waveforms_obj)
            return audio.transpose(0, 1).squeeze(1)

        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # Default stop to start -> if not specified, num_frames becomes 0,
        # which is the torchaudio default
        stop = waveforms_obj.get("stop", start)
        num_frames = stop - start
        audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
        if num_frames<self.audio_length*16000:
            padding_frames=self.audio_length*16000-num_frames
            padding_zeros=torch.zeros(padding_frames).unsqueeze(0)
            if start*16000-padding_frames<0:
                audio = torch.cat((padding_zeros, audio),dim=1)
            else:
                audio = torch.cat((audio, padding_zeros),dim=1)

        if audio.shape[1] == 160001:
            audio = audio[:, :160000]
        audio = audio.transpose(0, 1)
        return audio.squeeze(1)

    def __getitem__(self, idx):
        #print(self.data_keys[idx])
        entry = self.data_json[self.data_keys[idx]]
        start_time = float(self.data_keys[idx].split("_")[-2])
        waveform = self.read_audio(entry["wav"])
        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase}"
        if self.apply_1_shot:
            pre_speech_prompt += '''Use the following array format for the output: e.g., [|infant cry|,|infant cry|,|infant| |cry|;|female singing|,|female singing|,,,,,,]'''

        pre_speech_prompt += "\n\nInput:\n<speech>"
        post_speech_prompt = f"</speech>\n\n" + \
             "Output:\n"
        out_dict = {}
        output_prompt = "{["

        out_array=[""]*int(self.target_length*10)
        for key, value in entry["label"].items():
            #print(key,value)
            spk = self.spk_dict[key.split("_")[0]]
            voc = key.split("_")[1]
            if spk.startswith("irrelevant") or spk=="child":
                voc="speech"
            elif voc in self.voc_dict:
                voc=self.voc_dict[voc]
            else:
                continue
            for start,end in value:
                start_interval, end_interval = round(start-start_time,1),round(end-start_time,1)
                start_idx, end_idx = int(round(start_interval*10,1)), int(round(end_interval*10,1))
                for i in range(start_idx,end_idx):
                    if out_array[i]!="":
                        out_array[i]+=";"
                    out_array[i]+="|"+spk+"|"+" "+"|"+voc+"|"

        for i in range(len(out_array)):
            if out_array[i]=="":
                output_prompt+=","
            else:
                output_prompt+=out_array[i]+","
        output_prompt = output_prompt.rstrip('\n')[:-1] + "]}"

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]

class LBAudioDatasetMiddleFrame:
    def __init__(self, tokenizer, json_file, apply_1_shot=False, mode='train', audio_length=5, inference_only=False):

        f = open(json_file,"r")
        self.tokenizer = tokenizer
        self.data_json = json.load(f)
        self.data_keys = list(self.data_json.keys())
        self.spk_dict={"CHN":"infant","FAN":"female","MAN":"male","CXN":"child","FAN2":"irrelevant_female","MAN2":"irrelevant_male"}
        self.voc_dict={"FUS":"fussing","CRY":"crying","BAB":"babbling",
                "CDS":"child_directed_speech", "PLA":"child_directed_speech", "PLAC":"child_directed_speech",\
                "FAN":"adult_directed_speech", "MAN":"adult_directed_speech", "LAU":"laughter", "LAUC":"laughter",\
                "SNG":"singing", "SNGC":"singing"}
        
        ###changed by Nur
        self.common_phrases_set = "DO NOT OUTPUT ANY EXPLANATION OR ANYTHING OTHER THAN THE OUTPUT VOCALIZATION AS A PYTHON LIST IN THE PARTICULAR FORMAT."
        
        self.instruction_phrases_set_train = [
            "Figure out who's talking and what kind of sounds they're making in the middle frame in the audio.",
            "Identify the speaker and the types of vocalizations in the central frame based on the given sound input.",
            "Determine who the speaker is and what vocal sounds they produce in the center frame from the audio provided.",
            "Spot the person speaking and the nature of their vocalizations in the middle frame using the audio input.",
            "Recognize the speaker and the kinds of vocal sounds in the central frame with the supplied audio.",
            "Ascertain who is speaking and the type of vocalizations in the center frame based on the audio data.",
            "Find out the speaker and the variety of vocal sounds in the middle frame using the audio provided.",
            "Detect who’s talking and the specific vocalizations in the central frame from the given audio.",
            "Identify the individual speaking and their vocalization types in the center frame with the provided audio input.",
            "Determine the person speaking and the kind of vocal sounds in the middle frame based on the audio.",
            "Recognize who's talking and the types of vocalizations in the central frame using the audio input.",
            "Spot the speaker and the nature of their vocal sounds in the center frame from the given audio.",
            "Figure out who is speaking and the vocalization types in the middle frame using the provided audio data.",
            "Identify the person talking and the types of vocal sounds in the central frame based on the audio input.",
            "Determine who's speaking and what vocalizations they are making in the center frame with the given audio.",
            "Detect the individual speaking and the kinds of vocal sounds in the middle frame using the audio provided.",
            "Recognize the speaker and their vocalization types in the central frame based on the audio data.",
            "Ascertain who's talking and the nature of their vocalizations in the center frame from the provided audio.",
            "Find out who the speaker is and the types of vocal sounds in the middle frame using the given audio.",
            "Spot the person speaking and the specific vocalizations in the central frame with the supplied audio input.",
            "Identify who’s talking and what kind of vocal sounds they’re producing in the center frame based on the audio.",
            "Determine the speaker and the variety of vocalizations in the middle frame from the provided audio.",
            "Recognize who is speaking and the types of vocal sounds in the central frame using the audio input.",
            "Detect the individual talking and the nature of their vocalizations in the center frame with the given audio.",
            "Figure out who the speaker is and the types of vocal sounds in the middle frame based on the audio provided.",
            "Identify who’s speaking and the specific vocalization types in the central frame using the audio data.",
            "Determine the person talking and the kinds of vocal sounds in the center frame from the supplied audio.",
            "Spot who is speaking and the variety of vocalizations in the middle frame using the provided audio input.",
            "Recognize the individual talking and their vocalization types in the central frame based on the audio.",
            "Ascertain who’s the speaker and what vocal sounds they’re making in the center frame with the given audio.",
            "Identify who is speaking and what sounds they are making in the middle section of the audio.",
        ]

        self.instruction_phrases_set_valid = [
            "Detect who the person is and the kinds of vocal sounds in the central frame using the audio input.",
            "Identify the speaker and the nature of their vocalizations in the center frame based on the audio provided.",
            "Determine who’s talking and the types of vocal sounds they produce in the middle frame with the given audio.",
            "Recognize who is speaking and the variety of vocalizations in the central frame using the provided audio.",
            "Spot the individual talking and the types of vocal sounds in the center frame from the audio input.",
            "Figure out the speaker and the nature of their vocalizations in the middle frame based on the provided audio.",
            "Identify who's speaking and what kind of vocal sounds they're making in the central frame using the audio data.",
            "Determine the person speaking and the types of vocalizations in the center frame with the given audio.",
            "Detect who’s talking and the specific kinds of vocal sounds in the middle frame from the provided audio.",
            "Recognize the speaker and the variety of vocalizations in the central frame based on the audio input.",
        ]

        # optionlly 10 instructions for test set
        # self.instruction_phrases_set_test = [
        #     "Ascertain who is speaking and the types of vocal sounds in the center frame using the provided audio.",
        #     "Find out who’s the person talking and the nature of their vocalizations in the middle frame from the audio.",
        #     "Spot who the speaker is and what types of vocal sounds they’re producing in the central frame with the given audio.",
        #     "Identify who is talking and the kinds of vocalizations in the center frame based on the audio provided.",
        #     "Determine the speaker and the types of vocal sounds they make in the middle frame using the audio input.",
        #     "Recognize who’s speaking and the nature of their vocalizations in the central frame from the supplied audio.",
        #     "Detect the individual speaking and the variety of vocal sounds in the center frame with the given audio.",
        #     "Figure out who’s talking and the types of vocalizations in the middle frame based on the provided audio data.",
        #     "Identify the person speaking and the specific vocal sounds in the central frame using the audio input."
        #     "Find out who's speaking and the types of vocalizations in the middle frame from the provided audio.",
        # ]

        self.instruction_phrases_set_test = [
            "Detect the individual speaking and the variety of vocal sounds in the center frame with the given audio."
        ]

        self.instruction_phrases_set_test = [
            "Detect the individual speaking and the variety of vocal sounds in the center frame with the given audio."
        ]

        self.instruction_phrases_set={"train": self.instruction_phrases_set_train, \
                                      "valid": self.instruction_phrases_set_valid,\
                                      "test": self.instruction_phrases_set_test
                                      }
        self.apply_1_shot = apply_1_shot
        self.instruction_phrases=self.instruction_phrases_set[mode]
        self.audio_length=audio_length
        self.inference_only=inference_only

    def __len__(self):
        return len(self.data_keys)

    def read_audio(self, waveforms_obj):
        if isinstance(waveforms_obj, str):
            audio, _ = torchaudio.load(waveforms_obj)
            return audio.transpose(0, 1).squeeze(1)

        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # Default stop to start -> if not specified, num_frames becomes 0,
        # which is the torchaudio default
        stop = waveforms_obj.get("stop", start)
        num_frames = stop - start
        audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
        if num_frames<self.audio_length*16000:
            padding_frames=self.audio_length*16000-num_frames
            padding_zeros=torch.zeros(padding_frames).unsqueeze(0)
            if start*16000-padding_frames<0:
                audio = torch.cat((padding_zeros, audio),dim=1)
            else:
                audio = torch.cat((audio, padding_zeros),dim=1)
        if audio.shape[1] == 480001: ##changed
            audio = audio[:, :480000]

        audio = audio.transpose(0, 1)
        return audio.squeeze(1)
    def __getitem__(self, idx):
        #print(self.data_keys[idx])
        entry = self.data_json[self.data_keys[idx]]
        start_time = float(self.data_keys[idx].split("_")[-2])
        waveform = self.read_audio(entry["wav"])
        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"<audio>\n{instruction_phrase}"
        if self.apply_1_shot:
            pre_speech_prompt += '''Return output in a signle element list and separate overlapped vocalization by;'''

        # pre_speech_prompt += "\n\nInput:\n<speech>"
        post_speech_prompt = f"</speech>\n\n" + \
             "Output:\n"
        out_dict = {}
        output_prompt=""
        if not self.inference_only:
            output_prompt = "["

            out_string=""
            for spk in ["chn","fan","man","cxn","fan2","man2"]:
                # if spk=="fan2":
                #     spk="fan"
                # if spk=="man2":
                #     spk="man"
                voc_name=None
                if entry[spk]!="":
                    spk_name=self.spk_dict[spk.upper()]
                    if entry[spk] in self.voc_dict:
                        voc_name=self.voc_dict[entry[spk]]
                    elif spk in ["cxn", "fan2", "man2"]:
                        voc_name="speech"
                    if voc_name is not None:
                        # out_string+="<"+spk_name+"-"+voc_name+">;"  
                        out_string+= spk_name+"-"+voc_name+";"
                # else:
                #     out_string+="no sound;"   
                        # out_string+=spk_name+";"      
            out_string=out_string[:-1]
            output_prompt += out_string + "]"
            if output_prompt == '[]':
                output_prompt = '[No_sound]'
        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]
    # def __getitem__(self, idx):
    #     #print(self.data_keys[idx])
    #     entry = self.data_json[self.data_keys[idx]]
    #     start_time = float(self.data_keys[idx].split("_")[-2])
    #     waveform = self.read_audio(entry["wav"])
    #     instruction_phrase = random.choice(self.instruction_phrases)

    #     pre_speech_prompt = f"{instruction_phrase}"
    #     if self.apply_1_shot:
    #         pre_speech_prompt += '''Return output in a signle element list and separate overlapped vocalization by ;'''

        
    #     # pre_speech_prompt += "\n\nInput:\n<speech>"
    #     # post_speech_prompt = f"</speech>\n\n" + \
    #     #      "Output:\n"
    #     ###changed
        
    #     out_dict = {}

    #     output_prompt=""
    #     if not self.inference_only:
    #         output_prompt = ""

    #         out_string=""
    #         for spk in ["chn","fan","man","cxn","fan2","man2"]:
    #             voc_name=None
    #             if entry[spk]!="":
    #                 spk_name=self.spk_dict[spk.upper()]
    #                 if entry[spk] in self.voc_dict:
    #                     voc_name=self.voc_dict[entry[spk]]
    #                 elif spk in ["cxn", "fan2", "man2"]:
    #                     voc_name="speech"
    #                 if voc_name is not None:
    #                     out_string+=spk_name+"-"+voc_name+";"                
    #         out_string=out_string[:-1]

    #         output_prompt += out_string 
    #         # output_prompt = post_speech_prompt+output_prompt
    #     complete_prompt = pre_speech_prompt + output_prompt
        
    #     message = [{'role': 'user', 'content': pre_speech_prompt}, {'role': 'assistant', 'content': output_prompt}]
    #     chat_message =  self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    #     input_ids = self.tokenizer(chat_message, add_special_tokens=False).input_ids
    #     # print(len(input_ids[0]), len(input_ids[1]))
    #     # input_ids = torch.tensor(input_ids, dtype=torch.long)
    #     target_ids = copy.deepcopy(input_ids)
    #     instruction = self.tokenizer.apply_chat_template(message[:1], tokenize=False, add_generation_prompt=True)
    #     conversation = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    #     instruction_len = len(self.tokenizer(instruction, add_special_tokens=False).input_ids)
    #     conversation_len = len(self.tokenizer(conversation, add_special_tokens=False).input_ids)
    #     # print(chat_message)
    #     # chat_message = copy.deepcopy(message)
    #     # chat_message=self.tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=False)
    #     input_ids= torch.tensor(input_ids, dtype=torch.long)
    #     target_ids = torch.tensor(target_ids, dtype=torch.long)
    #     return waveform, input_ids, target_ids, instruction_len, self.data_keys[idx]
        # return waveform, pre_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]
# class MyCollatorFrame:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, batch):
#         # Separate the batch into components
#         waveforms = [item[0].unsqueeze(0) for item in batch]
#         input_ids = [item[1] for item in batch]
#         target_ids = [item[2] for item in batch]
#         instruction_len = [item[3] for item in batch]
#         data_ids = [item[4] for item in batch]
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels,
#                                                  batch_first=True,
#                                                  padding_value=IGNORE_INDEX)
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
#         labels = labels[:, :self.tokenizer.model_max_length]

#         input_ids = self.tokenizer(message, add_special_tokens=False, padding="longest", truncation=True, return_tensors='pt').input_ids
#         targets = copy.deepcopy(input_ids[-1])

        
        
#         # need fix if does not work
#         # torch.tensor(input_ids, dtype=torch.long)
        
        
#         # Stack waveforms into a batch
#         waveforms = torch.cat(waveforms, dim=0)

#         return waveforms, message, data_ids

class MyCollatorSpeakerEmbedding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Separate the batch into components
        waveforms = [item[0].unsqueeze(0) for item in batch]
        enroll_ids = [item[1] for item in batch]
        pre_speech_prompts = [item[2] for item in batch]
        
        fan_speech_prompt = [item[3] for item in batch]
        man_speech_prompt = [item[4] for item in batch]
        cxn_speech_prompt = [item[5] for item in batch]

        post_speech_prompts = [item[6] for item in batch]
        output_prompts = [item[7] for item in batch]
        data_ids = [item[9] for item in batch]

        # Tokenize prompts
        pre_tokenized_ids = self.tokenizer(
            pre_speech_prompts, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]
        
        fan_tokenized_ids = self.tokenizer(
            fan_speech_prompt, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]

        man_tokenized_ids = self.tokenizer(
            man_speech_prompt, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]

        cxn_tokenized_ids = self.tokenizer(
            cxn_speech_prompt, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]

        post_tokenized_ids = self.tokenizer(
            post_speech_prompts, 
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]

        output_tokenized_ids = self.tokenizer(
            [self.tokenizer.bos_token + prompt + self.tokenizer.eos_token for prompt in output_prompts],
            padding="longest", 
            return_tensors='pt', 
            truncation=False, 
            add_special_tokens=False
        )["input_ids"]
        
        # Stack waveforms into a batch
        waveforms = torch.cat(waveforms, dim=0)

        return waveforms, enroll_ids, pre_tokenized_ids, fan_tokenized_ids, man_tokenized_ids, cxn_tokenized_ids, post_tokenized_ids, output_tokenized_ids, data_ids

class LBAudioDatasetSpeakerEmbeddingFrameBased:
    def __init__(self, json_file, apply_1_shot=False, mode='train', audio_length=5, target_length=1):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        f = open(json_file,"r")
        self.data_json = json.load(f)
        self.data_keys = list(self.data_json.keys())
        self.spk_dict={"CHN":"infant","FAN":"female","MAN":"male","CXN":"child","FAN2":"irrelevant female","MAN2":"irrelevant male"}
        self.voc_dict={"FUS":"fussing","CRY":"crying","BAB":"babbling",
                "CDS":"child-directed speech", "PLA":"child-directed speech", "PLAC":"child-directed speech",\
                "FAN":"adult-directed speech", "MAN":"adult-directed speech", "LAU":"laughter", "LAUC":"laughter",\
                "SNG":"singing", "SNGC":"singing"}
        
        self.instruction_phrases_set_train = [
            "Identify who's speaking and the types of sounds they're making in each frame using the enrolled speaker data and the audio input.",
            "Determine the speaker and their vocalization categories at each frame level with the provided speaker embeddings and audio.",
            "Figure out who the speaker is and the kinds of vocal sounds in every frame using the enrolled embeddings and the audio input.",
            "Spot the person talking and the types of vocalizations per frame based on the enrolled speaker embeddings and audio data.",
            "Recognize the speaker and the variety of vocal sounds in each frame using the given speaker embeddings and audio input.",
            "Find out who’s speaking and the types of vocalizations frame by frame with the enrolled speaker embeddings and the audio.",
            "Detect the individual speaking and their vocalization types at each frame level using the provided speaker embeddings and audio input.",
            "Identify the person talking and the kinds of vocal sounds in every frame based on the enrolled embeddings and audio data.",
            "Determine who is speaking and the types of vocalizations for each frame using the enrolled speaker data and audio input.",
            "Figure out the speaker and their vocalization types per frame with the given speaker embeddings and audio.",
            "Spot who's talking and the variety of vocal sounds in each frame using the enrolled embeddings and audio input.",
            "Recognize the individual speaking and their types of vocalizations at frame level based on the speaker embeddings and audio.",
            "Find out who the speaker is and the kinds of vocal sounds in every frame with the provided speaker embeddings and audio input.",
            "Detect who’s speaking and the types of vocalizations for each frame using the enrolled speaker data and audio.",
            "Identify the speaker and their vocalization categories frame by frame with the given speaker embeddings and audio input.",
            "Determine the person talking and the types of vocal sounds in each frame based on the enrolled embeddings and audio data.",
            "Figure out who is speaking and the variety of vocalizations at each frame level using the provided speaker embeddings and audio.",
            "Spot the individual speaking and their types of vocal sounds per frame with the enrolled speaker data and audio input.",
            "Recognize who's talking and the kinds of vocalizations in every frame using the given speaker embeddings and audio.",
            "Find out who the speaker is and the types of vocal sounds for each frame based on the enrolled embeddings and audio input.",
            "Detect the person speaking and their vocalization types frame by frame with the provided speaker embeddings and audio.",
            "Identify who’s speaking and the variety of vocalizations at each frame level using the enrolled speaker data and audio input.",
            "Determine the speaker and their types of vocal sounds in every frame based on the enrolled embeddings and audio.",
            "Figure out who the individual is and the types of vocalizations per frame using the provided speaker embeddings and audio input.",
            "Spot who’s talking and the kinds of vocal sounds at each frame level with the enrolled speaker data and audio.",
            "Recognize the person speaking and their vocalization types in every frame using the given speaker embeddings and audio input.",
            "Find out who is speaking and the variety of vocal sounds for each frame based on the enrolled embeddings and audio.",
            "Detect who the speaker is and the types of vocalizations frame by frame with the provided speaker embeddings and audio.",
            "Identify the individual talking and their types of vocal sounds at each frame level using the enrolled speaker data and audio input.",
        ]

        self.instruction_phrases_set_valid = [
            "Determine who’s speaking and the kinds of vocalizations in every frame based on the enrolled embeddings and audio.",
            "Figure out the speaker and their vocalization types for each frame with the given speaker embeddings and audio input.",
            "Spot who is talking and the variety of vocal sounds at each frame level using the enrolled speaker data and audio.",
            "Recognize who the speaker is and the types of vocalizations in every frame based on the provided speaker embeddings and audio.",
            "Find out who’s speaking and the kinds of vocal sounds per frame with the enrolled embeddings and audio input.",
            "Detect the person talking and their vocalization types in each frame using the provided speaker data and audio.",
            "Identify who is speaking and the variety of vocal sounds frame by frame based on the enrolled embeddings and audio input.",
            "Determine the speaker and their types of vocalizations at each frame level using the given speaker embeddings and audio.",
            "Figure out who’s talking and the kinds of vocal sounds in every frame with the enrolled speaker data and audio input.",
            "Spot the individual speaking and their vocalization types per frame based on the provided speaker embeddings and audio.",
            "Recognize who is speaking and the variety of vocal sounds for each frame using the enrolled embeddings and audio input."
            "Find out the speaker and their types of vocalizations in each frame based on the provided speaker embeddings and audio."
        ]

        # optionlly 10 instructions for test set
        # self.instruction_phrases_set_test = [
        #     "Detect who’s speaking and the kinds of vocal sounds frame by frame with the enrolled speaker data and audio input.",
        #     "Identify the person talking and their vocalization types at each frame level using the given speaker embeddings and audio.",
        #     "Determine who is speaking and the variety of vocal sounds in every frame based on the enrolled embeddings and audio input.",
        #     "Figure out who the speaker is and the types of vocalizations per frame using the provided speaker data and audio.",
        #     "Spot who’s talking and the kinds of vocal sounds at each frame level with the enrolled speaker embeddings and audio input.",
        #     "Recognize the individual speaking and their types of vocalizations in every frame based on the enrolled embeddings and audio.",
        #     "Find out who is speaking and the variety of vocal sounds for each frame using the given speaker data and audio input.",
        #     "Detect the speaker and their types of vocalizations frame by frame with the provided speaker embeddings and audio.",
        #     "Identify who’s speaking and the kinds of vocal sounds at each frame level using the enrolled embeddings and audio input.",
        #     "Determine the person talking and the variety of vocalizations in every frame based on the provided speaker data and audio.",
        #     "Figure out who is speaking and the types of vocal sounds per frame with the enrolled speaker embeddings and audio input."
        # ]

        self.instruction_phrases_set_test = [
            "Detect who’s speaking and the kinds of vocal sounds frame by frame with the enrolled speaker data and audio input.", 
        ]

        self.instruction_phrases_set={"train": self.instruction_phrases_set_train, \
                                      "valid": self.instruction_phrases_set_valid,\
                                      "test": self.instruction_phrases_set_test
                                      }
        self.apply_1_shot = apply_1_shot
        self.instruction_phrases=self.instruction_phrases_set[mode]
        self.audio_length=audio_length
        self.target_length=target_length

    def __len__(self):
        return len(self.data_keys)

    def read_audio(self, waveforms_obj):
        if isinstance(waveforms_obj, str):
            audio, _ = torchaudio.load(waveforms_obj)
            return audio.transpose(0, 1).squeeze(1)

        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # Default stop to start -> if not specified, num_frames becomes 0,
        # which is the torchaudio default
        stop = waveforms_obj.get("stop", start)
        num_frames = stop - start
        audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
        if audio.shape[1] == 32001:
            audio = audio[:, :32000]
        audio = audio.transpose(0, 1)
        return audio.squeeze(1)

    def __getitem__(self, idx):
        #print(self.data_keys[idx])
        entry = self.data_json[self.data_keys[idx]]
        start_time = float(self.data_keys[idx].split("_")[-2])
        waveform = self.read_audio(entry["wav"])

        # find family id
        curr_id = self.data_keys[idx]
        site_name = "_".join(curr_id.split("_")[:2])

        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase}"
        if self.apply_1_shot:
            pre_speech_prompt += '''Use the following array format for the output: e.g., [|infant cry|,|infant cry|,|infant| |cry|;|female singing|,|female singing|,,,,,,]'''

        pre_speech_prompt += "\n\nInput:\n<speech>"
        fan_speech_prompt = "</speech>\n\n<female_enroll>"
        man_speech_prompt = "</female_enroll>\n\n<male_enroll>"
        cxn_speech_prompt = "</male_enroll>\n\n<child_enroll>"
        post_speech_prompt = "</child_enroll>\n\n" + "Output:\n"
        out_dict = {}
        output_prompt = "{["

        out_array=[""]*int(self.target_length*10)
        for key, value in entry["label"].items():
            #print(key,value)
            spk = self.spk_dict[key.split("_")[0]]
            voc = key.split("_")[1]
            if spk.startswith("irrelevant") or spk=="child":
                voc="speech"
            elif voc in self.voc_dict:
                voc=self.voc_dict[voc]
            else:
                continue
            for start,end in value:
                start_interval, end_interval = round(start-start_time,1),round(end-start_time,1)
                start_idx, end_idx = int(round(start_interval*10,1)), int(round(end_interval*10,1))
                for i in range(start_idx,end_idx):
                    if out_array[i]!="":
                        out_array[i]+=";"
                    out_array[i]+="|"+spk+"|"+" "+"|"+voc+"|"

        for i in range(len(out_array)):
            if out_array[i]=="":
                output_prompt+=","
            else:
                output_prompt+=out_array[i]+","
        output_prompt = output_prompt.rstrip('\n')[:-1] + "]}"

        complete_prompt = pre_speech_prompt + fan_speech_prompt + man_speech_prompt + cxn_speech_prompt + post_speech_prompt + output_prompt
        return waveform, site_name, pre_speech_prompt, fan_speech_prompt, man_speech_prompt, cxn_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]

class LBAudioSpeakerEmbeddingDataset:
    def __init__(self, json_file, apply_1_shot=False, mode='train'):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        f = open(json_file,"r")
        self.data_json = json.load(f)
        self.data_keys = list(self.data_json.keys())
        self.spk_dict={"CHN":"infant","FAN":"female","MAN":"male","CXN":"child","FAN2":"irrelevant female","MAN2":"irrelevant male"}
        self.voc_dict={"FUS":"fussing","CRY":"crying","BAB":"babbling",
                "CDS":"child-directed speech", "PLA":"child-directed speech", "PLAC":"child-directed speech",\
                "FAN":"adult-directed speech", "MAN":"adult-directed speech", "LAU":"laughter", "LAUC":"laughter",\
                "SNG":"singing", "SNGC":"singing"}
                
        self.instruction_phrases_set_train = [
            "Begin by counting the vocalized segments in the audio, then identify the start and end times for the speaker and type of vocalization using the enrolled speaker embeddings.",
            "First, tally the vocalized sections in the audio, then pinpoint the starting and ending times for both the speaker and the vocalization type, guided by enrolled speaker embeddings.",
            "Count the vocalized parts in the audio first, then use the provided speaker embeddings to detect when each segment begins and ends, as well as the vocalization type.",
            "Start by determining how many vocalized segments are present in the audio, then find the timestamps for the speaker and vocalization type using the enrolled speaker embeddings.",
            "Identify the total number of vocalized segments in the audio, and then locate the start and end times for each speaker and vocalization type with the help of the enrolled embeddings.",
            "First, figure out how many vocalized segments exist in the audio, and then detect when each segment starts and ends, along with its vocalization type, using the enrolled speaker embeddings.",
            "Count the vocal segments in the recording first, and then use enrolled speaker embeddings to mark the starting and ending times of each speaker and vocalization type.",
            "First, calculate the number of vocalized portions in the audio, and then determine the timestamps for the speaker and type of vocalization using the provided embeddings.",
            "Begin by assessing how many vocalized segments are in the audio, then identify their start and stop times along with the speaker and vocalization type using enrolled speaker embeddings.",
            "Determine the count of vocalized audio segments first, and then use speaker embeddings to find the starting and ending times for the speaker and type of vocalization for each segment.",
            "Start by counting how many vocalized segments are present, then utilize enrolled speaker embeddings to mark the beginning and ending timestamps for each speaker and vocalization type.",
            "First, identify the number of vocalized sections in the audio, then detect their timestamps and the vocalization type with the aid of enrolled speaker embeddings.",
            "Count the vocalized segments in the audio first, and then determine when each segment starts and ends, including the vocalization type, using the speaker embeddings provided.",
            "First, calculate the total number of vocalized segments, and then pinpoint the start and end times for the speaker and vocalization type using the given enrolled embeddings.",
            "Count the number of segments in the audio where vocalization occurs, then use enrolled speaker embeddings to identify the timestamps and vocalization type for each one.",
            "Begin by figuring out how many vocalized parts are in the audio, and then use the enrolled embeddings to detect the starting and ending times for each speaker and vocalization type.",
            "First, determine the count of vocalized portions in the recording, and then find the start and end times for the speaker and vocalization type based on the enrolled speaker embeddings.",
            "Calculate the number of vocalized segments in the audio first, then locate their timestamps and determine the vocalization type using the enrolled speaker embeddings.",
            "First, figure out the total number of vocalized sections, and then detect when each segment begins and ends, identifying the vocalization type with enrolled speaker embeddings.",
            "Count the vocalized audio segments first, then pinpoint the start and end times for each speaker and vocalization type using the provided enrolled speaker embeddings.",
            "Start by counting the number of vocalized segments in the recording, and then find the timestamps for the speaker and vocalization type with the help of the enrolled embeddings.",
            "Determine how many vocalized segments are present, then use the enrolled speaker embeddings to identify the start and end times and the vocalization type for each one.",
            "First, identify the number of vocalized parts in the audio, then locate the start and stop times for each segment and its vocalization type using the given embeddings.",
            "Calculate the number of vocalized sections first, then use enrolled speaker embeddings to pinpoint the starting and ending timestamps for each speaker and vocalization type.",
            "Start by assessing the total vocalized segments, then detect their start and end times along with the vocalization type using the provided enrolled embeddings.",
            "First, count the segments in the audio that contain vocalization, and then determine the timestamps and vocalization type for each one based on the speaker embeddings.",
            "Figure out the total number of vocalized segments in the recording, and then use the enrolled speaker embeddings to detect the starting and ending times for the vocalization type.",
            "Begin by calculating the number of vocalized portions, and then locate their start and end times while identifying the speaker and type of vocalization using the embeddings.",
            "Count the vocal segments in the audio, and then detect the timestamps and vocalization type for each segment with the help of enrolled speaker embeddings.",
            "First, identify the vocalized portions in the recording, then use the provided embeddings to determine when each segment starts and ends, as well as the vocalization type.",
        ]
        
        self.instruction_phrases_set_valid = [
            "Start by calculating how many vocalized segments there are in the audio, then use the enrolled speaker embeddings to pinpoint their timestamps and identify the vocalization type.",
            "First, determine the count of vocalized audio segments, and then find the starting and ending times for the speaker and vocalization type with the enrolled speaker embeddings.",
            "Count the number of vocalized segments first, and then use speaker embeddings to mark the timestamps and identify the vocalization type for each segment.",
            "First, calculate the vocalized parts in the audio, and then determine when each segment starts and ends, along with the vocalization type, using the enrolled speaker embeddings.",
            "Figure out how many vocalized portions are in the audio first, then use the provided embeddings to detect the timestamps and vocalization type for each segment.",
            "Begin by counting the total vocalized sections, then locate their start and end times while identifying the vocalization type and speaker with the enrolled speaker embeddings.",
            "First, assess the number of vocalized parts in the recording, and then find the timestamps and vocalization type for each segment based on the provided embeddings.",
            "Count the vocalized segments in the audio first, and then detect the timestamps and vocalization type for each part with the help of enrolled speaker embeddings.",
            "Start by identifying the number of vocalized segments, then use the provided embeddings to find the starting and ending times and identify the vocalization type.",
            "First, figure out how many vocalized sections exist in the audio, and then pinpoint their timestamps and vocalization type using the enrolled speaker embeddings.",
        ]
        
        # optinally 10 instructions for test set
        # self.instruction_phrases_set_test = [
        #     "Calculate the number of vocalized segments in the recording first, then use the provided embeddings to locate the timestamps and vocalization type for each one.",
        #     "Start by counting how many vocalized parts are present in the audio, then identify their start and end times and vocalization type using enrolled speaker embeddings.",
        #     "First, identify the total number of vocalized segments, and then determine when each segment begins and ends using the speaker embeddings to guide the process.",
        #     "Count the vocalized portions of the audio, then use the provided enrolled embeddings to detect their timestamps and the type of vocalization for each one.",
        #     "First, calculate the number of vocalized audio segments, then identify the starting and ending times for the speaker and vocalization type with the enrolled speaker embeddings.",
        #     "Figure out the total vocalized segments in the recording, then use the enrolled speaker embeddings to determine the timestamps and vocalization type for each part.",
        #     "Begin by counting how many vocalized parts are present in the audio, then find the start and end times for each segment and its vocalization type using the embeddings.",
        #     "First, determine the number of vocalized sections, then use enrolled speaker embeddings to identify their start and end times and classify their vocalization type.",
        #     "Count the vocalized segments in the recording, then use the provided embeddings to locate the timestamps and vocalization type for each one."
        # ]

        # use 1 instruction for test set
        self.instruction_phrases_set_test = [
            "First, calculate the number of vocalized audio segments, then identify the starting and ending times for the speaker and vocalization type with the enrolled speaker embeddings.",
        ]


        self.instruction_phrases_set={"train": self.instruction_phrases_set_train, \
                                      "valid": self.instruction_phrases_set_valid,\
                                      "test": self.instruction_phrases_set_test
                                      }

        self.instruction_phrases=self.instruction_phrases_set[mode]
        
    def __len__(self):
        return len(self.data_keys)

    def read_audio(self, waveforms_obj):
        if isinstance(waveforms_obj, str):
            audio, _ = torchaudio.load(waveforms_obj)
            return audio.transpose(0, 1).squeeze(1)

        path = waveforms_obj["file"]
        start = waveforms_obj.get("start", 0)
        # Default stop to start -> if not specified, num_frames becomes 0,
        # which is the torchaudio default
        stop = waveforms_obj.get("stop", start)
        num_frames = stop - start
        audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
        if audio.shape[1] == 32001:
            audio = audio[:, :32000]
        audio = audio.transpose(0, 1)
        return audio.squeeze(1)

    def __getitem__(self, idx):
        #print(self.data_keys[idx])
        entry = self.data_json[self.data_keys[idx]]
        start_time = float(self.data_keys[idx].split("_")[-2])
        waveform = self.read_audio(entry["wav"])

        # find family id
        curr_id = self.data_keys[idx]
        if curr_id.startswith("e"):
            file_name="_".join(curr_id.split("_")[:3])[1:]
            site_name=self.lena_id_map[file_name]
        else:
            site_name = "_".join(curr_id.split("_")[:2])

        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase}"
        if self.apply_1_shot:
            if self.include_voc_count:
                pre_speech_prompt += '''Use the following json format for the output: e.g., {"number of vocalization": 2, "|infant| |babbling|": [0.3, 1.0], "|female| |child-directed speech|":[2.0,3.5]}.'''
            else:
                pre_speech_prompt += '''Use the following json format for the output: e.g., {"|infant| |babbling|": [0.3, 1.0], "|female| |child-directed speech|":[2.0,3.5]}.'''

        pre_speech_prompt += "\n\nInput:\n<speech>"
        fan_speech_prompt = "</speech>\n\n<female_enroll>"
        man_speech_prompt = "</female_enroll>\n\n<male_enroll>"
        cxn_speech_prompt = "</male_enroll>\n\n<child_enroll>"
        post_speech_prompt = "</child_enroll>\n\n" + "Output:\n"
        out_dict = {}
        output_prompt = "{"

        curr_count = 0
        for key, value in entry["label"].items():
            #print(key,value)
            spk = self.spk_dict[key.split("_")[0]]
            voc = key.split("_")[1]
            if spk.startswith("irrelevant") or spk=="child":
                voc="speech"
            elif voc in self.voc_dict:
                voc=self.voc_dict[voc]
            else:
                continue
            for start,end in value:
                if "|"+spk+"|"+" "+"|"+voc+"|" not in out_dict:
                    out_dict["|"+spk+"|"+" "+"|"+voc+"|"]=[]
                out_dict["|"+spk+"|"+" "+"|"+voc+"|"].append((round(start-start_time,1),round(end-start_time,1)))
                curr_count+=1

        #if self.include_voc_count:
        output_prompt +=f'  "number of vocalization": {curr_count}, '

        for key,value in out_dict.items():
            for start,end in value:
                output_prompt +=f'  "{key}": [{start},{end}], '

        output_prompt = output_prompt.rstrip(',\n') + "}"

        complete_prompt = pre_speech_prompt + fan_speech_prompt + man_speech_prompt + cxn_speech_prompt + post_speech_prompt + output_prompt
        return waveform, site_name, pre_speech_prompt, fan_speech_prompt, man_speech_prompt, cxn_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, self.data_keys[idx]

# Example usage
from tqdm import tqdm
if __name__ == "__main__":
    dev_file="/work/hdd/bebr/PRJ_LLM_SP25/data/nur_json_file/train_30s_real.json"
    # out_path = "train_snr_10_mix_8_5s_with_prompt.json"
    # dataset = LBAudioDatasetSpeakerEmbeddingFrameBased(
    #             json_file=dev_file, 
    #             mode='test',
    #             apply_1_shot=True,
    #             audio_length=2,
    #             target_length=2,
    #            )
    dataset = LBAudioDatasetMiddleFrame(
                json_file=dev_file, 
                mode='train',
                apply_1_shot=False,
                # include_voc_count=True,
                audio_length=30,
               )
    
    # dataset = LBAudioDataset(
    #         json_file=dev_file,
    #         mode='train',
    #         apply_1_shot=True,
    #         include_voc_count=True,
    #     )

    # with open(dev_file, "r") as f:
    #     data = json.load(f)
    # items_list = list(my_dict.items())
    # print(len(dataset))
    # count =0
    # unique_dict={}
    # text_file = "event_10s_train_real.txt"
    # f = open(text_file, 'w')
    # for idx, (key, value) in enumerate(data.items()):
    #     waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, data_key = dataset[idx]
    #     value["prompt"] = output_prompt
    # with open(out_path, "w") as f:
    #     json.dump(data, f, indent=4)
    for i in tqdm(range(len(dataset))):
        # print(len(dataset[i]))
        #waveform, enrollment, pre_speech_prompt, fan_speech_prompt, man_speech_prompt, cxn_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, data_ids = dataset[i]
        # waveform, site_name, pre_speech_prompt, fan_speech_prompt, man_speech_prompt, cxn_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, data_ids= dataset[i] ##for speaker_emebedding
        waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, data_key = dataset[i] ##for 0.1s frame_based
        # print(waveform.shape)
        if waveform.shape[0]!=480000 and waveform.shape[0]!=480001:
            print(waveform.shape)
            print(data_key)
        # waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt, data_keys[idx]
        

        # print(output_prompt)
        # f.write(output_prompt + '\n')
        # print(data.items()[0])
        # exit()
    #     key, value = items_list[i]
    #     value["prompt"] = output_prompt   # or any default string you want
        
        # if output_prompt in unique_dict:
        #     unique_dict[output_prompt]+=1
        # else:
        #     unique_dict[output_prompt]=1
        
        # print(dkey)
        # print(pre_speech_prompt)
        # print('------------')
        # print(post_speech_prompt)
        # print('--------')
        # print(output_prompt)
        # print(complete_prompt)
        # exit()
        # count+=1
        # if count>1000:
        #     break
        
        # if waveform.shape[0]>32000:
        #     # print(waveform.shape)
        #     count +=1
    
    # print(unique_dict)