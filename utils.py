import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F 
import math, sys
from collections import defaultdict
import tensorly as tl
from functools import partial
from _TTLoRAWrapper_TensorMultiplication import TTLoRALinearWrapper_withcores, AdaptCores_and_Test_Individual

# custom_cache_dir = "/lustre/vescratch1/ceodspspectrum/cache_hf/"
# os.environ['HF_DATASETS_CACHE'] = custom_cache_dir

def get_ttlora_shape(ttlora_shape_from_config):
    ttlora_shape = ttlora_shape_from_config
    return ttlora_shape

def get_ttlora_rank(r, ttlora_shape):
    ttlora_rank = [1]
    for i in range(len(ttlora_shape)-1):
        ttlora_rank.append(r)
    ttlora_rank.append(1)
    return ttlora_rank

def load_new_model_for_sequence_classification_from_local_path(config):
    model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=2)
    # if "llama-3.1-8b" in config["model_name"] or "llama-3.1-70b" in config["model_name"]:
    #     model.config.pad_token_id = model.config.eos_token_id[0]
    model.config.pad_token_id = model.config.eos_token_id
    for param in model.parameters():
        param.requires_grad = False
    # print(model)
    return model

def get_tokenizer(config, dataset):
    '''Tokenizes the provided dataset and data name using the tokenizer from the specified path'''
    path = config["tokenizer_path"]
    data_name = config["dataset_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("EOS Token ID at tokenizer:", tokenizer.eos_token_id, type(tokenizer.eos_token_id))
    tokenizer.pad_token = tokenizer.eos_token
    print("EOS Token at tokenizer:", tokenizer.eos_token, type(tokenizer.eos_token))
    
    def tokenize_text(batch):
        # Truncation true = truncate the tokenized text to max_length
        # Padding true = pad the tokenized text to max_length
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "rte":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "mnli":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "stsb":
        #     return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "wsc":
        #     return tokenizer(batch["text"], batch['span1_text'], batch['span2_text'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "winogrande":
        #     return tokenizer(batch["sentence"], batch['option1'], batch['option2'], batch['answer'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "ax":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "multirc":
        #     return tokenizer(batch["paragraph"], batch['question'], batch['answer'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "boolq":
        #     return tokenizer(batch["question"], batch['passage'], add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "hellaswag":
        #     return tokenizer(batch["ind"], batch["activity_label"], batch['ctx_a'],batch['ctx_b'],batch['ctx'],batch['endings'],batch['source_id'],batch['split'],batch['split_type'], add_special_tokens=True, truncation=True, padding=True)
        
        #super_glue datasets
        if data_name == "wic":
            # Properly formatting inputs into a single string
            texts = [f'{word} {s1} {s2} {s1_start}-{s1_end} {s2_start}-{s2_end}'
                 for word, s1, s2, s1_start, s1_end, s2_start, s2_end in 
                 zip(batch["word"], batch["sentence1"], batch["sentence2"], batch["start1"], batch["end1"], batch["start2"], batch["end2"])]

            return tokenizer(texts, add_special_tokens=True, truncation=True, padding=True)
        # if data_name == "cb":
        #     return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding=True)
        if data_name == "boolq":
            return tokenizer(batch["question"], batch["passage"],  truncation=True, padding=True)
        # if data_name == "wsc":
        #     target = batch["target"]
        #     span1_text=[]
        #     span2_text=[]

        #     for entry in target:
        #         span1_text_val=entry.get('span1_text')
        #         span2_text_val=entry.get('span2_text')

        #         span1_text.append(span1_text_val)
        #         span2_text.append(span2_text_val)
        #     span = [f'{span1_text}{tokenizer.bos_token}{span2_text}' for span1_text, span2_text in zip(span1_text, span2_text)]
        #     return tokenizer(batch["text"], span, truncation=True, padding=True,max_length = 1024)
        # if data_name == "copa":
        #     question = batch["question"]
        #     choice1 = batch["choice1"]
        #     choice2 = batch["choice2"]
        #     choice2 = batch["question"]
        #     combined = [f'{choice1}{tokenizer.bos_token}{choice2}{tokenizer.bos_token}{question}' for choice1, choice2, question in zip(choice1, choice2, question)]
        #     return tokenizer(batch["premise"], combined, truncation=True, padding=True)
    # Map the words in the dataset to the token values of the loaded tokenizer
    # None batch size = process entire dataset as single batch
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 

    ### change the format into tensors of the specific columns
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def get_mix_tokenizer(model_name, path, data_name, dataset): #used to do the maximum padding for the dataset to match all the types of dataset's sequence_length
    '''Tokenizes the provided dataset and data name using the tokenizer from the specified path'''
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    if "roberta" in model_name:
        max_context_length = 512
    elif "llama" in model_name:
        max_context_length = 1024
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
    def tokenize_text(batch):
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "rte":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True,  padding="max_length", max_length=max_context_length)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True,  padding="max_length", max_length=max_context_length)
        # if data_name == "mnli":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "stsb":
        #     return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "wsc":
        #     return tokenizer(batch["text"], batch['span1_text'], batch['span2_text'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "winogrande":
        #     return tokenizer(batch["sentence"], batch['option1'], batch['option2'], batch['answer'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "ax":
        #     return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        # if data_name == "multirc":
        #     return tokenizer(batch["paragraph"], batch['question'], batch['answer'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "boolq":
            return tokenizer(batch["question"], batch['passage'], add_special_tokens=True, truncation=True,  padding="max_length", max_length=max_context_length)
        if data_name == "wic":
            # Properly formatting inputs into a single string
            texts = [f'{word} {s1} {s2} {s1_start}-{s1_end} {s2_start}-{s2_end}'
                 for word, s1, s2, s1_start, s1_end, s2_start, s2_end in 
                 zip(batch["word"], batch["sentence1"], batch["sentence2"], batch["start1"], batch["end1"], batch["start2"], batch["end2"])]

            return tokenizer(texts, add_special_tokens=True, truncation=True,  padding="max_length", max_length=max_context_length)
        
        # if data_name == "wsc":
        #     target = batch["target"]
        #     span1_text=[]
        #     span2_text=[]

        #     for entry in target:
        #         span1_text_val=entry.get('span1_text')
        #         span2_text_val=entry.get('span2_text')

        #         span1_text.append(span1_text_val)
        #         span2_text.append(span2_text_val)
        #     span = [f'{span1_text}{tokenizer.bos_token}{span2_text}' for span1_text, span2_text in zip(span1_text, span2_text)]
        #     return tokenizer(batch["text"], span, truncation=True, padding='max_length', max_length = 1024)
        # if data_name == "copa":
        #     question = batch["question"]
        #     choice1 = batch["choice1"]
        #     choice2 = batch["choice2"]
        #     choice2 = batch["question"]
        #     combined = [f'{choice1}{tokenizer.bos_token}{choice2}{tokenizer.bos_token}{question}' for choice1, choice2, question in zip(choice1, choice2, question)]
        #     return tokenizer(batch["premise"], combined, truncation=True, padding='max_length', max_length = 1024)
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def load_mixed_datasets(model_name, dataset_names, tokenizer_path):
    '''Dataset loading and check if loaded correctly'''
    mixed_train_dataset_dict = {
        
        "input_ids": torch.empty(0,dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    mixed_validation_dataset_dict = {
        "input_ids": torch.empty(0, dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    for dataset_name in dataset_names:
        take_train = 2490 #for rte 2490, mrpc 3668
        take_val = 277 #for rte 277, mrpc 408
        print("Loading dataset inside mixed datasets: ", dataset_name)
        if dataset_name in ["mrpc", "cola", "sst2", "qnli", "rte", "qqp"]:
            glue_type = "glue"
        elif dataset_name in ["boolq", "wic"]:
            glue_type = "super_glue"

        # dataset = load_dataset(glue_type,dataset_name)
        dataset = load_dataset_(dataset_name)
        tokenized = get_mix_tokenizer(model_name, tokenizer_path, dataset_name , dataset)
        train_tokenized_dataset = tokenized["train"]
        train_tokenized_dataset = train_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Train tokenized dataset before slicing: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)

        # print("Train tokenized dataset: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)
        validation_tokenized_dataset = tokenized["validation"]
        validation_tokenized_dataset = validation_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Validation tokenized dataset before slicing: ", validation_tokenized_dataset['input_ids'].shape, validation_tokenized_dataset['attention_mask'].shape, validation_tokenized_dataset['label'].shape)

        #########################################For Train###################################################
        mixed_train_dataset_dict["input_ids"] = torch.cat((mixed_train_dataset_dict["input_ids"], 
                                                           train_tokenized_dataset["input_ids"][:take_train]), dim=0)
        mixed_train_dataset_dict["attention_mask"] = torch.cat((mixed_train_dataset_dict["attention_mask"], 
                                                                train_tokenized_dataset["attention_mask"][:take_train]), dim=0)
        mixed_train_dataset_dict["label"] = torch.cat((mixed_train_dataset_dict["label"], 
                                                       train_tokenized_dataset["label"][:take_train]), dim=0)
        #########################################For Validation###################################################

        mixed_validation_dataset_dict["input_ids"] = torch.cat((mixed_validation_dataset_dict["input_ids"], 
                                                                validation_tokenized_dataset["input_ids"][:take_val]), dim=0)
        mixed_validation_dataset_dict["attention_mask"] = torch.cat((mixed_validation_dataset_dict["attention_mask"], 
                                                                     validation_tokenized_dataset["attention_mask"][:take_val]), dim=0)
        mixed_validation_dataset_dict["label"] = torch.cat((mixed_validation_dataset_dict["label"], 
                                                            validation_tokenized_dataset["label"][:take_val]), dim=0)
    
    # print("mixed_train_dataset_dict: ", 
    #       mixed_train_dataset_dict['input_ids'].shape, 
    #       mixed_train_dataset_dict['attention_mask'].shape, 
    #       mixed_train_dataset_dict['label'].shape,
    #       "Data types: ", 
    #       mixed_train_dataset_dict['input_ids'].dtype, 
    #       mixed_train_dataset_dict['attention_mask'].dtype, 
    #       mixed_train_dataset_dict['label'].dtype
    #       )
    # print("mixed_validation_dataset_dict: ", 
    #       mixed_validation_dataset_dict['input_ids'].shape, 
    #       mixed_validation_dataset_dict['attention_mask'].shape, 
    #       mixed_validation_dataset_dict['label'].shape,
    #       "Data types: ", 
    #       mixed_validation_dataset_dict['input_ids'].dtype, 
    #       mixed_validation_dataset_dict['attention_mask'].dtype, 
    #       mixed_validation_dataset_dict['label'].dtype
    #       )
    # print(mixed_train_dataset_dict['label'].shape, mixed_train_dataset_dict['label'])
    # Shuffle the training dataset
    train_indices = torch.randperm(mixed_train_dataset_dict["input_ids"].size(0))
    mixed_train_dataset_dict["input_ids"] = mixed_train_dataset_dict["input_ids"][train_indices]
    mixed_train_dataset_dict["attention_mask"] = mixed_train_dataset_dict["attention_mask"][train_indices]
    mixed_train_dataset_dict["label"] = mixed_train_dataset_dict["label"][train_indices]

    # Shuffle the validation dataset
    val_indices = torch.randperm(mixed_validation_dataset_dict["input_ids"].size(0))
    mixed_validation_dataset_dict["input_ids"] = mixed_validation_dataset_dict["input_ids"][val_indices]
    mixed_validation_dataset_dict["attention_mask"] = mixed_validation_dataset_dict["attention_mask"][val_indices]
    mixed_validation_dataset_dict["label"] = mixed_validation_dataset_dict["label"][val_indices]
    
    return mixed_train_dataset_dict, mixed_validation_dataset_dict

def wrap_model_with_ttcores(model, config):
    
    ttlora_shape_q = get_ttlora_shape(config["qshape"])
    ttlora_rank_q = get_ttlora_rank(config["rank"], ttlora_shape_q)
    ttlora_shape_v = get_ttlora_shape(config["vshape"])
    ttlora_rank_v = get_ttlora_rank(config["rank"], ttlora_shape_v)

    m_factors_q = config["m_factors_q"]
    n_factors_q = config["n_factors_q"]
    m_factors_v = config["m_factors_v"]
    n_factors_v = config["n_factors_v"]

    ttlora_alpha = config["alpha"]
    ttlora_adapter_at_query = True
    ttlora_adapter_at_value = True

    
    assign_ttlora = partial(TTLoRALinearWrapper_withcores, alpha=ttlora_alpha)

    if "roberta" in config["model_name"]:
        for layer in model.roberta.encoder.layer:
            if ttlora_adapter_at_query:
                layer.attention.self.query = assign_ttlora(layer.attention.self.query,
                                                           tt_shape=ttlora_shape_q, 
                                                           tt_rank=ttlora_rank_q,
                                                           m_factors=m_factors_q,
                                                           n_factors=n_factors_q,
                                                           device=config["device"], 
                                                           init_choice=config["core_init_choice"]) 
            if ttlora_adapter_at_value:
                layer.attention.self.value = assign_ttlora(layer.attention.self.value,
                                                           tt_shape=ttlora_shape_v, 
                                                           tt_rank=ttlora_rank_v,
                                                           m_factors=m_factors_v,
                                                           n_factors=n_factors_v,
                                                           device=config["device"],
                                                           init_choice=config["core_init_choice"]) 
    elif "llama" in config["model_name"]:
        for layer in model.model.layers:
            if ttlora_adapter_at_query:
                layer.self_attn.q_proj = assign_ttlora(layer.self_attn.q_proj,
                                                       tt_shape=ttlora_shape_q, 
                                                       tt_rank=ttlora_rank_q,
                                                       m_factors=m_factors_q,
                                                       n_factors=n_factors_q,
                                                       device=config["device"],
                                                       init_choice=config["core_init_choice"])
            if ttlora_adapter_at_value:
                layer.self_attn.v_proj = assign_ttlora(layer.self_attn.v_proj,
                                                       tt_shape=ttlora_shape_v,
                                                       tt_rank=ttlora_rank_v,
                                                       m_factors=m_factors_v,
                                                       n_factors=n_factors_v,
                                                       device=config["device"],
                                                       init_choice=config["core_init_choice"])
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
    return model

def parse_experts(directory_path, model_name, dataload_type, dataset_name, multiple_datasets):

    """
    Parses all `.ckpt` files inside expert subfolders and organizes the experts into a nested dictionary.
    Saves ttlora cores and classifier weights for each expert.
    """
    # Nested dictionary to hold all experts
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))
    # print(dataload_type, dataset_name, multiple_datasets)
    # Iterate through each expert folder in the directory
    if dataload_type == "multiple":
        expert_names = multiple_datasets
    elif dataload_type == "single":
        expert_names = dataset_name
    else:
        raise ValueError("Invalid dataload type. Please use 'single' or 'multiple' for dataload type.")
    # print(expert_names)
    for expert_name in expert_names:
        expert_folder = os.path.join(directory_path, expert_name)

        # Ensure it is a directory (expert folder)
        if os.path.isdir(expert_folder):
            # Iterate through .ckpt files inside the expert folder
            for filename in os.listdir(expert_folder):
                # Check if there are multiple .ckpt files in the expert folder
                ckpt_files = [f for f in os.listdir(expert_folder) if f.endswith(".ckpt")]
                if len(ckpt_files) > 1:
                    raise ValueError(f'Multiple .ckpt files found in {expert_folder}. Only one .ckpt file is allowed per expert folder.')
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(expert_folder, filename)
                    # Load the .ckpt file
                    checkpoint = torch.load(file_path, map_location="cpu")
                    # Extract model weights (state_dict)
                    expert_data = checkpoint["state_dict"] 
                    if "roberta" in model_name: 
                        expert_data = {k: v for k, v in expert_data.items() if 'tt_cores' in k or 'classifier' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'classifier' in parts:  #keys as: model.classifier.dense.weight
                                try:   
                                    classifier = parts[1]
                                    t_type = parts[2]
                                    w_b=parts[3]
                                    if t_type not in all_experts[expert_name][classifier]:
                                        all_experts[expert_name][classifier][t_type] = {}
                                    all_experts[expert_name][classifier][t_type][w_b] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:  #keys: model.roberta.encoder.layer.11.attention.self.query.ttlora_cores.0
                                try:
                                    layer = f'layer_{parts[4]}'  # Extract layer index
                                    attention_type = parts[7]  # 'query' or 'value'
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'tt_cores_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')
                    
                    elif "llama" in model_name:
                        expert_data = {k: v for k, v in expert_data.items() if 'tt_cores' in k or 'score' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'score' in parts:  #key as: model.score.weight (no bias)
                                try:   
                                    classifier = parts[1]
                                    w_b=parts[2]
                                    all_experts[expert_name][classifier][w_b] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:        #model.model.layers.1.self_attn.q_proj.ttlora_cores.7
                                try:
                                    layer = f'layer_{parts[3]}'  # Extract layer index
                                    attention_type = parts[5]  # 'query' or 'value'
                                    if attention_type == "q_proj":
                                        attention_type = "query"
                                    elif attention_type == "v_proj":
                                        attention_type = "value"
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'tt_cores_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')

    return all_experts

def parse_experts_for_single_test(directory_path, model_name):

    """
    Parses all `.ckpt` files inside expert subfolders and organizes the experts into a nested dictionary.
    Saves ttlora cores and classifier weights for each expert.
    """
    # Nested dictionary to hold all experts
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))

    # Iterate through each expert folder in the directory
    for expert_name in os.listdir(directory_path):
        expert_folder = os.path.join(directory_path, expert_name)

        # Ensure it is a directory (expert folder)
        if os.path.isdir(expert_folder):
            # Iterate through .ckpt files inside the expert folder
            for filename in os.listdir(expert_folder):
                # Check if there are multiple .ckpt files in the expert folder
                ckpt_files = [f for f in os.listdir(expert_folder) if f.endswith(".ckpt")]
                if len(ckpt_files) > 1:
                    raise ValueError(f'Multiple .ckpt files found in {expert_folder}. Only one .ckpt file is allowed per expert folder.')
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(expert_folder, filename)
                    # Load the .ckpt file
                    checkpoint = torch.load(file_path, map_location="cpu")
                    # Extract model weights (state_dict)
                    expert_data = checkpoint["state_dict"] 
                    if "roberta" in model_name: 
                        expert_data = {k: v for k, v in expert_data.items() if 'tt_core' in k or 'classifier' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'classifier' in parts:  #keys as: model.classifier.dense.weight
                                try:   
                                    classifier = parts[1]
                                    t_type = parts[2]
                                    w_b=parts[3]
                                    if t_type not in all_experts[expert_name][classifier]:
                                        all_experts[expert_name][classifier][t_type] = {}
                                    all_experts[expert_name][classifier][t_type][w_b] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:  #keys: model.roberta.encoder.layer.11.attention.self.query.ttlora_cores.0
                                try:
                                    layer = f'layer_{parts[4]}'  # Extract layer index
                                    attention_type = parts[7]  # 'query' or 'value'
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'tt_core_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')
                    
                    elif "llama" in model_name:
                        expert_data = {k: v for k, v in expert_data.items() if 'tt_core' in k or 'score' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'score' in parts:  #key as: model.score.weight (no bias)
                                try:   
                                    classifier = parts[1]
                                    w_b=parts[2]
                                    all_experts[expert_name][classifier][w_b] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:        #model.model.layers.1.self_attn.q_proj.ttlora_cores.7
                                try:
                                    layer = f'layer_{parts[3]}'  # Extract layer index
                                    attention_type = parts[5]  # 'query' or 'value'
                                    if attention_type == "q_proj":
                                        attention_type = "query"
                                    elif attention_type == "v_proj":
                                        attention_type = "value"
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'tt_core_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')

    return all_experts

def load_dataset_(data_name):
    path = "/usr/projects/unsupgan/afia/stack_v2"
    data_path = os.path.join(path, data_name)
    dataset = load_dataset(data_path)
    # dataset = load_dataset("glue", data_name)
    return dataset