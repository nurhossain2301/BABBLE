from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import torch

new_special_tokens = ["<infant-babbling>", "<infant-crying>", "<infant-fussing>", "<infant-laughter>", "<child-speech>", \
                      "<female-child_directed_speech>", "<female-laughter>", "female-singing>", "<female-adult_directed_speech>", \
                        "<male-adult_directed_speech>", "<male-child_directed_speech>", "<male-laughter>", "<irrelevant_female-child_directed_speech>", \
                        "<irrelevant_female-speech>", "<irrelevant_male-speech>", "<child-speech>", "<child-laughter>", "<child-child_directed_speech>", "<child-singing>", \
                        "<irrelevant_male-child_directed_speech>", "<irrelevant_male-adult_directed_speech>", "<irrelevant_female-adult_directed_speech>", "<male-singing>"]

def get_llm(name, use_lora, lora_r, lora_alpha, special_token=new_special_tokens, llm_model_path=None):
    # print('llm_model_name: ', name)
    # print('model_path: ', llm_model_path)
    llm_tokenizer = AutoTokenizer.from_pretrained(name)
    # llm_tokenizer.pre_tokenizer = Whitespace()

    llm_model = AutoModelForCausalLM.from_pretrained(
        name, 
        trust_remote_code=False,
        cache_dir = llm_model_path,
        # attn_implementation = "flash_attention_2"
        # device_map="auto",
        )
    # special_tokens_dict = {"additional_special_tokens": new_special_tokens}
    # llm_tokenizer.add_special_tokens(special_tokens_dict)
    # llm_model.resize_token_embeddings(len(llm_tokenizer))
    # token_id = llm_tokenizer.convert_tokens_to_ids("infant-crying")
    # print("Token ID:", token_id)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.padding_side = "right"
        # print('before:', len(llm_tokenizer))
        llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        llm_model.resize_token_embeddings(len(llm_tokenizer), mean_resizing=False)
        embedding_layer = llm_model.get_input_embeddings()
        emb_weights = embedding_layer.weight.data
        pad_id = llm_tokenizer.pad_token_id
        mask = torch.ones(emb_weights.size(0), dtype=torch.bool)
        mask[pad_id] = False  # exclude the pad token itself
        mean_vector = emb_weights[mask].mean(dim=0, keepdim=True)
        with torch.no_grad():
            embedding_layer.weight[pad_id] = mean_vector
        # print('after:', len(llm_tokenizer))
    

    if use_lora:
        peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules="all-linear",
                # target_modules=["lm_head"],
                # target_modules = ["q_proj","k_proj","v_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                bias="none"
            )

        llm_model = get_peft_model(llm_model, peft_config)
    
        
    
    for name, param in llm_model.named_parameters():
        if "lora_" not in name:       # only LoRA layers will have "lora_" in their names
            param.requires_grad = False
    
    # for param in llm_model.parameters():
    #     param.requires_grad = False
    llm_model.print_trainable_parameters()
    return llm_tokenizer, llm_model

if __name__ == "__main__":
    model = get_llm("TinyLlama/TinyLlama-1.1B-Chat-v1.0",True, 8, 16)
    # print(model + f'Do you want a joke or a poem? ' + gen(stop='.'))
    # print(model(f'Do you want a joke or a poem? '))