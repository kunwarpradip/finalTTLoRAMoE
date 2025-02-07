import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
import math
import sys
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset, Dataset
from torch import nn

#imports from local files
from model import CustomLightningModule
from utils import get_tokenizer, load_mixed_datasets, parse_experts, get_ttlora_shape, get_ttlora_rank
from utils import load_new_model_for_sequence_classification_from_local_path, load_dataset_
from _TTLoRAWrapper_TensorMultiplication import AdaptCores_and_Test_Individual, TTLoRALinearWrapper_withcores
from _TTMoE_multitask_wrapper import MoEsparseRouting, MoEsparseRoutingForClassification


tl.set_backend('pytorch')
# # Redirect stdout and stderr to a file
# sys.stdout = open('output.log', 'w')
# sys.stderr = open('output.log', 'w')


def apply_hooks(model):
    def forward_hook(module, input, output):
        if not hasattr(forward_hook, "called"):
            forward_hook.called = True
            print("*"*10,"Inside forward hook; Looking into grad and grad_fn whose grad_fn is not None","*"*10)
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if inp.grad_fn is not None:
                        print(f'{"Module Name", module.__class__.__name__}')
                        print(f'Shape: {inp.shape}, grad_fn: {inp.grad_fn}')
        else:
            if input.grad_fn is not None:
                print(f'{"Module Name", module.__class__.__name__}')
                print(f'Shape: {input.shape}, grad_fn: {input.grad_fn}')
        if isinstance(output, torch.Tensor):
            if output.grad_fn is not None:
                print(f'{"Module Name", module.__class__.__name__}')
                print(f'Shape: {output.shape}, grad_fn: {output.grad_fn}')
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if out.grad_fn is not None:
                        print(f'{"Module Name", module.__class__.__name__}')
                        print(f'Shape: {out.shape}, grad_fn: {out.grad_fn}')

    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)

    def backward_hook(module, grad_input, grad_output):
        if not hasattr(backward_hook, "called"):
            backward_hook.called = True
            print("*"*10,"Inside Backward hook; Looking into input and output grad at layers with requires_grad as True","*"*10)
        if any(param.requires_grad for param in module.parameters()):
            for i, grad in enumerate(grad_input):
                print(f'Module: {module.__class__.__name__}')
                print(f'Grad input: {grad}')
            for i, grad in enumerate(grad_output):
                print(f'Module: {module.__class__.__name__}')
                print(f'Grad Output: {grad}')
    
    for name, module in model.named_modules():
        module.register_full_backward_hook(backward_hook)

def train_moe_without_ray(config):
    
    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")

    '''Load the model and and define the labels'''
    model = load_new_model_for_sequence_classification_from_local_path(config)

    if "roberta" in config["model_name"]:
        model.roberta.encoder = MoEsparseRouting(model.roberta.encoder, config=config)
        
        model.classifier = MoEsparseRoutingForClassification(
                                base_classifier=model.classifier, 
                                config=config,
                                router=model.roberta.encoder,)
        
    elif "llama" in config["model_name"]:
        model.model = MoEsparseRouting(model.model, config=config)
        model.score = MoEsparseRoutingForClassification(
                                base_classifier=model.score, 
                                config=config,
                                router=model.model,)
    else:
        raise ValueError("Please provide the correct model name")

    for name, param in model.named_parameters():
        if "router" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    '''Dataset loading and check if loaded correctly'''
    if config["dataload_type"] == "single":
        # #For single dataset
        # dataset = load_dataset(config["glue_type"], config["dataset_name"])
        dataset = load_dataset_(config["dataset_name"])
        tokenized = get_tokenizer(config, dataset)
        train_dataset = tokenized["train"]
        val_dataset = tokenized["validation"]
    
    elif config["dataload_type"] == "multiple":
        #For multiple datasets
        train_dataset_dict, val_dataset_dict = load_mixed_datasets(config["model_name"],config["multiple_datasets"], config["tokenizer_path"])
        train_dataset = Dataset.from_dict(train_dataset_dict)
        val_dataset = Dataset.from_dict(val_dataset_dict)
        # print("train dataset feature types", type(train_dataset["input_ids"]), type(train_dataset["attention_mask"]), type(train_dataset["label"]))
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        print("Train dataset shapes: ", train_dataset['input_ids'].shape, train_dataset["input_ids"].dtype, train_dataset['attention_mask'].shape, train_dataset['label'].shape)
        print("Validation dataset shapes: ", val_dataset['input_ids'].shape, val_dataset['attention_mask'].shape, val_dataset['label'].shape)

    else:
        raise ValueError("Please provide the correct dataload type")

    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,  # 32 for llama, 256 for roberta
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=8, #16 for llama, 8 for roberta
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
       )

    '''For trainig and evaluation'''
    lightning_model = CustomLightningModule(model, 
                                            config)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
        )

    if config["dataload_type"] == "single":
        dirpath = f'./checkpoints/{config["model_name"]}/moe/single/{config["dataset_name"]}'
    elif config["dataload_type"] == "multiple":
        dirpath = f'./checkpoints/{config["model_name"]}/moe/multiple/{'_'.join(config["multiple_datasets"])}'
    else:
        raise ValueError("Please provide the correct dataload type")

    '''Callback provided by PyTorch Lightning that allows to save model checkpoints during training'''
    model_checkpoint_callback=ModelCheckpoint(
        dirpath=dirpath,
        save_top_k=1, 
        mode="max", 
        monitor="val_acc")  

    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        # enable_progress_bar=True,
        # enable_model_summary=False, 
        # log_every_n_steps=10,
    )
    
    start = time.time()
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                #checkpoint_path is used to load the model from the checkpoint to resume the training
                # ckpt_path="./lightning_logs/version_2/checkpoints/epoch=0-step=819.ckpt"
                )
    end = time.time()
    elapsed = end - start
    print(f'Time elapsed {elapsed/60:.2f} min')

    '''Model Testing in test and validation datasets'''
    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc=trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    print("-"*50, "\nTraining Accuracy: ", train_acc, "\nValidation Accuracy: ", val_acc)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)


    return{ "Model_name": config["model_name"],
            "Model_path": config["model_path"],
            "Tokenizer_path": config["tokenizer_path"],
            "Taining_Accuracy": train_acc[0]['accuracy'], 
            "Validation_Accuray": val_acc[0]['accuracy'],
            "Router Type": config["router_type"],
            "Trainable_parameters_count": train_params,
            "epochs": trainer.current_epoch,
            "model": config["model_name"],
            "dataload_type": config["dataload_type"],
            "multiple_datasets": config["multiple_datasets"],
            "Query Shape": config["qshape"],
            "Query m_factors": config["m_factors_q"],
            "Query n_factors": config["n_factors_q"],
            "Value Shape": config["vshape"],
            "Value m_factors": config["m_factors_v"],
            "Value n_factors": config["n_factors_v"],
            "common_alpha": config["common_alpha"],
            "learning_rate": config["learning_rate"]
            }

def main():
    
    #changeable model parameter
    #changeable model parameter
    model_name = "llama-3.2-1b" # options: roberta-base, llama-3.2-1b, llama-3.2-3b, llama-3.1-8b, llama-3.1-70b, 
    dataload_type= "multiple" # {single, multiple}
    dataset_name = "mrpc" # if dataload_type single, this goes if multiple then multiple datasets goes in 
    #{for glue: mrpc, cola, sst2, qnli}, {for super_glue: boolq, wic}
    multiple_datasets= ["cola", "mrpc","rte","wic"] # combination of the datasets/experts both to be used
    experts_dict = parse_experts(f'./trained_checkpoints/{model_name}/experts/', model_name, dataload_type, dataset_name, multiple_datasets)
    #check conditions
    if not experts_dict:
        raise ValueError("The experts dictionary is empty. Please provide valid experts.")
    if model_name not in ["roberta-base", "llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.1-70b"]:
        (lambda: (_ for _ in ()).throw(ValueError(f'{model_name}: Not adapted for this model')))()
    if dataset_name in ["mrpc", "cola", "sst2", "qnli", "rte", "qqp"]:
        glue_type = "glue"
    elif dataset_name in ["boolq", "wic"]:
        glue_type = "super_glue"
    else:
        (lambda: (_ for _ in ()).throw(ValueError(f'{dataset_name}: Not adapted for this dataset')))(),
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        #ttlora parameters
        #query parameters
        "qshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta query shape = 768x768
        else [16,4,4,2,2,2,2,2,2,4,4,16] if "llama-3.2-1b" in model_name #llama-3.2-1b q_proj shape = 2048x2048
        else [16,4,4,3,2,2,2,2,3,4,4,16] if "llama-3.2-3b" in model_name #llama-3.2-3b q_proj shape = 3072x3072
        else [16,4,4,4,2,2,2,2,4,4,4,16] if "llama-3.1-8b" in model_name #llama-3.1-8b q_proj shape = 4096x4096,
        else [16,4,4,4,2,2,2,2,2,2,4,4,4,16] if "llama-3.1-70b" in model_name #llama-3.1-70b q_proj shape = 8192x8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "m_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta m of query shape = 768,
        else [16,4,4,2,2,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of q_proj shape = 2048
        else [16,4,4,3,2,2] if "llama-3.2-3b" in model_name #llama-3.2-3b m of q_proj shape = 3072
        else [16,4,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b m of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b m of q_proj shape = 8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "n_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta n of query shape = 768
        else [16,4,4,2,2,2] if "llama-3.2-1b" in model_name #llama-3.2-1b n of q_proj shape = 2048
        else [16,4,4,3,2,2] if "llama-3.2-3b" in model_name #llama-3.2-3b n of q_proj shape = 3072,
        else [16,4,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b n of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b n of q_proj shape = 8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        #value parameters
        "vshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta value shape = 768x768
        else [16,4,4,2,2,2,2,4,4,16] if "llama-3.2-1b" in model_name #llama-3.2-1b v_proj shape = 2048 x 512
        else [16,4,4,3,2,2,2,2,4,4,16] if "llama-3.2-3b" in model_name #llama-3.2-3b v_proj shape = 3072x1024
        else [16,4,4,4,2,2,2,2,4,4,16] if "llama-3.1-8b" in model_name #llama-3.1-8b v_proj shape = 4096x1024,
        else [16,4,4,4,2,2,2,2,2,4,4,16] if "llama-3.1-70b" in model_name #llama-3.1-70b v_proj shape = 8192x1024
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "m_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta m of value shape = 768
        else [16,4,4,2,2,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of v_proj shape = 2048
        else [16,4,4,3,2,2] if "llama-3.2-3b" in model_name #llama-3.2-3b m of v_proj shape = 3072,
        else [16,4,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b m of v_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b m of v_proj shape = 8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "n_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta n of value shape = 768
        else [16,4,4,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of v_proj shape = 512
        else [16,4,4,2,2] if "llama-3.2-3b" in model_name #llama-3-3b n of v_proj shape = 1024
        else [16,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b n of v_proj shape = 1024
        else [16,4,4,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b n of v_proj shape = 1024
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        
        "rank": 
        4 if "roberta-base" in model_name 
        else 10 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "alpha": 
        8 if "roberta-base" in model_name 
        else 12 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),
        
        "common_alpha": 
        8 if "roberta-base" in model_name 
        else 12 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        #model parameters
        "model_name" : model_name,
        "model_path" : '/lustre/vescratch1/ceodspspectrum/llms/llama31-8b/checkpoints/', #for local
        "tokenizer_path" : '/lustre/vescratch1/ceodspspectrum/llms/llama31-8b/checkpoints/', #for local
        # "model_path" : f'./models/{model_name}/{model_name}-model', 
        # "tokenizer_path" : f'./models/{model_name}/{model_name}-tokenizer', 
        "device": device,  
  
        #changable dataset parameters:
        "glue_type": glue_type,
        "dataload_type": dataload_type,
        "dataset_name" : dataset_name, 
        "multiple_datasets": multiple_datasets, 

        #experts and moe parameters
        "experts_dict": experts_dict,
        
        #hyperparameters
        "router_type" : "attention", # {single_layer, multi_layer, attention}
        "gumbel_temperature": 1.0,
        "learning_rate": 1e-5,
    }

    analysis =  train_moe_without_ray(config)
    # df = pd.DataFrame(list(analysis.items()), columns=['metric', 'value'])
    # print(df)
    # if config["dataload_type"] == "single":
    #     filename = f'MoE_{config["dataload_type"]}_{config["dataset_name"]}_{config["model_name"]}.csv'
    # elif config["dataload_type"] == "multiple":
    #     filename = f'MoE_{config["dataload_type"]}{'_'.join(config["multiple_datasets"])}_{config["model_name"]}.csv'
    # else:
    #     print("Please provide the correct dataload type")
    #     sys.exit()
    # df.to_csv(filename, index=False)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main()