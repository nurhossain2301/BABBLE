import torch
from torch import nn
import torch.nn.functional as F
import fairseq
import os
# from connector import get_connector
from peft import get_peft_model, LoraConfig, TaskType


from transformers import AutoModel, WhisperProcessor, WhisperModel

class FairseqWav2Vec2(nn.Module):
    """This lobe enables the integration of fairseq pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    FairSeq >= 1.0.0 needs to be installed:
    https://fairseq.readthedocs.io/en/latest/

    The model can be used as a fixed features extractor or can be finetuned. It
    will download automatically the model if a url is given (e.g FairSeq
    repository from GitHub).

    Arguments
    ---------
    pretrained_path : str
        Path of the pretrained wav2vec2 model. It can be a url or a local path.
    save_path : str
        Path and filename of the downloaded model.
    input_norm : bool (default: None)
        If True, a layer_norm (affine) will be applied to the input waveform.
        By default, it is extracted from the checkpoint of the downloaded model
        in order to match the pretraining conditions. However, if this information
        is not given in the checkpoint, it has to be given manually.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.
    dropout : float (default: None)
        If different from None (0.0 to 1.0), it will override the given fairseq
        dropout rates. This is useful if the wav2vec2 model has been trained
        without dropout and one wants to reactivate it for downstream task
        fine-tuning (better performance observed).

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
    >>> save_path = "models_checkpoints/wav2vec2.pt"
    >>> model = FairseqWav2Vec2(model_url, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 100,  768])
    """

    def __init__(
        self,
        save_path,
        input_norm=None,
        output_norm=True,
        freeze=True,
        pretrain=True,
        dropout=None,
        encoder_dropout = 0.0, 
        output_all_hiddens=True,
        tgt_layer=None,
        include_CNN_layer=False,
    ):
        super().__init__()

        # # Download the pretrained wav2vec2 model. It can be local or online.
        # if not os.path.exists(save_path):
        #     download_file(pretrained_path, save_path)

        # During pretraining dropout might be set to 0. However, we might want
        # to apply dropout when fine-tuning on a downstream task. Hence we need
        # to modify the fairseq cfg to activate dropout (if requested).
        print(save_path)
        overrides={}
        if encoder_dropout is not None:
            overrides = {
                "model": {
                    "encoder_layerdrop": encoder_dropout,
                }
            }
        if not freeze:
            if dropout is not None and encoder_dropout is not None:
                overrides = {
                    "model": {
                        "dropout": dropout,
                        "encoder_layerdrop": encoder_dropout,
                        "dropout_input": dropout,
                        "attention_dropout": dropout,
                    }
                }
            elif dropout is not None:
                overrides = {
                    "model": {
                        "dropout": dropout,
                        "dropout_input": dropout,
                        "attention_dropout": dropout,
                    }
                }     
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [save_path], arg_overrides=overrides
        )

        # wav2vec pretrained models may need the input waveform to be normalized
        # Hence, we check if the model has be trained with or without it.
        # If the information isn't contained in the checkpoint IT HAS TO BE GIVEN
        # BY THE USER.
        if input_norm is None:
            if hasattr(cfg["task"], "normalize"):
                self.normalize = cfg["task"].normalize
            elif hasattr(cfg, "normalize"):
                self.normalize = cfg.normalize
            else:
                self.normalize = False
        else:
            self.normalize = input_norm

        model = model[0]
        self.model = model
        self.freeze = freeze

        self.output_norm = output_norm

        if self.freeze:
            self.model.eval()
            # Freeze parameters
            for param in model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in model.parameters():
                param.requires_grad = True

        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            self.reset_layer(self.model)

        # Following the fairseq implementation of downstream training,
        # we remove some modules that are unnecessary.
        self.remove_pretraining_modules()
        self.output_all_hiddens = output_all_hiddens
        self.tgt_layer = tgt_layer
        self.include_CNN_layer=include_CNN_layer

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Extracts the wav2vect embeddings"""
        # We normalize the input signal if needed.
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        if self.tgt_layer=="CNN": #initial embeddings from conv
            out = self.model.extract_features(wav, padding_mask=None, mask=False)
            out = self.model.post_extract_proj(out['features'])
        elif isinstance(self.tgt_layer, int):
            out = self.model.extract_features(wav, padding_mask=None, mask=False, layer=self.tgt_layer)['x']
        else: 
            out = self.model.extract_features(wav, padding_mask=None, mask=False, layer=self.tgt_layer)
            if self.output_all_hiddens or isinstance(self.tgt_layer, list):
                out = self.aggregate_features(out) # 12, B, T, D
                if isinstance(self.tgt_layer, list):
                    out = out[self.tgt_layer]
            else:
                out = out['x']
                
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out
    
    def aggregate_features(self, out):
        features = []
        for i in range(len(out['layer_results'])):
            curr_feature = out['layer_results'][i][0].transpose(0,1)
            features.append(curr_feature)
        features = torch.stack(features)
        return features


    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)

    def remove_pretraining_modules(self):
        """ Remove uneeded modules. Inspired by the same fairseq function."""

        self.model.quantizer = None
        self.model.project_q = None
        self.model.target_glu = None
        self.model.final_proj = None

class WhisperAudioEncoder(nn.Module):
    def __init__(self, 
                model_name='openai/whisper-large-v2', 
                model_path=None,
                output_all_hiddens=False,
                finetune=False):
        super().__init__()
        lora_r = 4
        lora_alpha = 8
        lora_target_modules = ["q_proj", "v_proj"]
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=model_path)
        self.model = WhisperModel.from_pretrained("openai/whisper-large-v2", cache_dir=model_path)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )
        base_model = get_peft_model(self.model, peft_config)
        
        base_model.model.load_state_dict(torch.load('/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/whisper_path/best_model_xulin.pt', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'], strict=False)
        self.encoder = base_model.encoder

        self.output_all_hiddens = output_all_hiddens

        for param in self.model.encoder.parameters():
            param.requires_grad = finetune
            
    def forward(self, x):
        input_features = self.processor.feature_extractor(x.cpu().numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(self.model.encoder.device)
        # print(input_features) ##check later
        encoder_outputs = self.model.encoder(input_features, output_hidden_states=self.output_all_hiddens)
        if self.output_all_hiddens:
            return torch.stack(encoder_outputs.hidden_states)[1:,:,:,:] # 33 x 1 x T x D
        else:
            return encoder_outputs.last_hidden_state
# class WhisperAudioEncoder(nn.Module):
#     def __init__(self, 
#                 model_name='openai/whisper-large-v2', 
#                 model_path=None,
#                 output_all_hiddens=True,
#                 finetune=False):
#         super().__init__()
#         self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", cache_dir=model_path)
#         self.model = WhisperModel.from_pretrained("openai/whisper-large-v2", cache_dir=model_path)
#         self.encoder = self.model.encoder
#         self.encoder.load_state_dict(torch.load('/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/whisper_path/best_model_xulin.pt', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'], strict=False)
#         self.output_all_hiddens = output_all_hiddens

#         #delete decoder part
#         # self.model.decoder.cpu()
#         # # Then we delete the decoder
#         # del self.model.decoder
#         # self.model.decoder = None

#         for param in self.model.encoder.parameters():
#             param.requires_grad = finetune
            
#     def forward(self, x):
#         input_features = self.processor.feature_extractor(x.cpu().numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(self.model.encoder.device)
#         # print(input_features) ##check later
#         encoder_outputs = self.model.encoder(input_features, output_hidden_states=self.output_all_hiddens)
#         if self.output_all_hiddens:
#             return torch.stack(encoder_outputs.hidden_states)[1:,:,:,:] # 33 x 1 x T x D
#         else:
#             return encoder_outputs.last_hidden_state
# class WhisperAudioEncoder(nn.Module):
#     def __init__(self, 
#                 model_name='openai/whisper-large-v3', 
#                 model_path=None,
#                 output_all_hiddens=True,
#                 finetune=False):
#         super().__init__()
#         self.processor = WhisperProcessor.from_pretrained(model_name, cache_dir = model_path)
#         self.model = WhisperModel.from_pretrained(model_name, cache_dir = model_path)
#         self.output_all_hiddens = output_all_hiddens

#         #delete decoder part
#         self.model.decoder.cpu()
#         # Then we delete the decoder
#         del self.model.decoder
#         self.model.decoder = None

#         for param in self.model.encoder.parameters():
#             param.requires_grad = finetune
            
#     def forward(self, x):
#         input_features = self.processor.feature_extractor(x.cpu().numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(self.model.encoder.device)
#         # print(input_features) ##check later
#         encoder_outputs = self.model.encoder(input_features, output_hidden_states=self.output_all_hiddens)
#         if self.output_all_hiddens:
#             return torch.stack(encoder_outputs.hidden_states)[1:,:,:,:] # 33 x 1 x T x D
#         else:
#             return encoder_outputs.last_hidden_state

def get_audio_encoder(name, finetune_encoder, model_path=None, output_all_hiddens=True):
    if name == 'wav2vec-LL4300':
        return FairseqWav2Vec2(model_path, freeze=(not finetune_encoder), output_all_hiddens=output_all_hiddens) 
    elif name.startswith("openai/whisper"):
        return WhisperAudioEncoder(name, model_path, finetune=finetune_encoder, output_all_hiddens=output_all_hiddens) 
    elif name == "merge":
        return FairseqWav2Vec2(model_path, freeze=(not finetune_encoder), output_all_hiddens=output_all_hiddens), WhisperAudioEncoder(name, model_path, finetune=finetune_encoder, output_all_hiddens=output_all_hiddens) 
    else:
        raise NotImplementedError

if __name__ == "__main__":
    # model = SpeechTokenizerEnoder()
    model = get_audio_encoder("wav2vec-LL4300", finetune_encoder=True, output_all_hiddens=False, model_path="/work/hdd/bebr/Models/MODEL_WEIGHTS/wav2vec_LL4300.pt")
    # print(model)

    x = torch.randn(2, 32000)
    connector1 = get_connector('linear-pool', 768, 2048, 10)
    z = model(x)
    z = connector1(z)
    print(z.shape)
