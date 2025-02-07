import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import sys
import copy
import warnings
import os

from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset
from model import CustomLightningModule
from utils import get_tokenizer, get_ttlora_shape, get_ttlora_rank, parse_experts, load_dataset_
from utils import load_new_model_for_sequence_classification_from_local_path, wrap_model_with_ttcores
from _TTLoRAWrapper_TensorMultiplication import TTLoRALinearWrapper_withcores, AdaptCores_and_Test_Individual

tl.set_backend('pytorch')

# Suppress all warnings
warnings.filterwarnings("ignore")

sys.stdout = open('output.log', 'w')
sys.stderr = open('output.log', 'w')

def train_without_ray(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")
    
    dataset = load_dataset(config["glue_type"], config["dataset_name"])
    # dataset = load_dataset_(config["dataset_name"])
    print("\nDataset type",config["dataset_name"], dataset)
    tokenized = get_tokenizer(config, dataset)
    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]
    #we don't use test datasets as they contain hidden labels as -1

    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,  # 32 for llama, 256 for roberta
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=8   #16 for llama, 8 for roberta
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        num_workers=8
        #no need to shuffle the validation data as to get the consistent evaluations
    )

    '''Load the model and and define the labels and makes the parameters untrainable'''
    model = load_new_model_for_sequence_classification_from_local_path(config)
    # print("Pad Token ID:", model.config.pad_token_id, type(model.config.pad_token_id))
    # sys.exit()
    wrapped_model = wrap_model_with_ttcores(model, config)

    lightning_model = CustomLightningModule(wrapped_model, config)
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
        )
    
    model_checkpoint_callback=ModelCheckpoint(
        # dirpath=f"./trained_checkpoints/{config["model_name"]}/experts/{config["dataset_name"]}",
        dirpath=f'./trained_checkpoints/{config["model_name"]}/experts/{config["dataset_name"]}/',
        save_top_k=1, 
        mode="max", 
        monitor="val_acc")  

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        log_every_n_steps=10,
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
    print(f"Time elapsed {elapsed/60:.2f} min")

    '''Evaluating the model on training and validation datasets'''
    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)

    print("-"*50, 
          "\nTraining Accuracy: ", train_acc, 
          "\nValidation Accuracy in best lightning model: ", val_acc)

    print("Best model path: ", model_checkpoint_callback.best_model_path)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)

    return {"Total_epochs": trainer.current_epoch + 1,
            "Taining_Accuracy": train_acc[0]['accuracy'], 
            "Validation_Accuray": val_acc[0]['accuracy'], 
            "Trainable_parameters_count": train_params,
            "Best_model_path": model_checkpoint_callback.best_model_path,
            "Query Shape": config["qshape"],
            "Query m_factors": config["m_factors_q"],
            "Query n_factors": config["n_factors_q"],
            "Value Shape": config["vshape"],
            "Value m_factors": config["m_factors_v"],
            "Value n_factors": config["n_factors_v"],
            "Rank": config["rank"],
            "Alpha": config["alpha"],
            "Learning_rate": config["learning_rate"],
            "Model_name": config["model_name"],
            "Model_path": config["model_path"],
            "Tokenizer_path": config["tokenizer_path"],
            "Dataset_Type": config["dataload_type"],
            "Dataset_name": config["dataset_name"],
            }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    
    #changeable model parameter
    model_name = "llama-3.2-1b" # options: roberta-base, llama-3.2-1b, llama-3-8b, llama-3-70b, 
    glue_type= "glue" # glue, super_glue
    dataset_name = "qqp" # glue are :mrpc, cola, sst2, qnli, rte, qqp;  super_glue are: boolq, cb, copa, wsc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        #ttlora parameters
        #query parameters
        "qshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta query shape = 768x768
        else [16,4,4,2,2,2,2,2,2,4,4,16] if "llama-3.2-1b" in model_name #llama-3.2-1b q_proj shape = 2048x2048
        else [16,4,4,4,2,2,2,2,4,4,4,16] if "llama-3-8b" in model_name #llama-3-8b q_proj shape = 4096x4096,
        else [16,4,4,4,2,2,2,2,2,2,4,4,4,16] if "llama-3-70b" in model_name #llama-3-70b q_proj shape = 8192x8192
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(),

        "m_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta m of query shape = 768,
        else [16,4,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b m of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3-70b" in model_name #llama-3-70b m of q_proj shape = 8192
        else [16,4,4,2,2,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of q_proj shape = 2048
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(), 

        "n_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta n of query shape = 768
        else [16,4,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b n of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3-70b" in model_name #llama-3-70b n of q_proj shape = 8192
        else [16,4,4,2,2,2] if "llama-3.2-1b" in model_name #llama-3.2-1b n of q_proj shape = 2048
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(),

        #value parameters
        "vshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta value shape = 768x768
        else [16,4,4,4,2,2,2,2,4,4,16] if "llama-3-8b" in model_name #llama-3-8b v_proj shape = 4096x1024,
        else [16,4,4,4,2,2,2,2,2,4,4,16] if "llama-3-70b" in model_name #llama-3-70b v_proj shape = 8192x1024
        else [16,4,4,2,2,2,2,4,4,16] if "llama-3.2-1b" in model_name #llama-3.2-1b n of v_proj shape = 2048 x 512
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(), 

        "m_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta m of value shape = 768
        else [16,4,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b m of v_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3-70b" in model_name #llama-3-70b m of v_proj shape = 8192
        else [16,4,4,2,2,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of v_proj shape = 2048
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(), 

        "n_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta n of value shape = 768
        else [16,4,4,2,2] if "llama-3-8b" in model_name #llama-3-8b n of v_proj shape = 1024
        else [16,4,4,2,2] if "llama-3-70b" in model_name #llama-3-70b n of v_proj shape = 1024
        else [16,4,4,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of v_proj shape = 512
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(),

        
        "rank": 
        4 if "roberta-base" in model_name 
        else 10 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(),

        "alpha": 
        8 if "roberta-base" in model_name 
        else 12 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(),
        
        "common_alpha": 
        8 if "roberta-base" in model_name 
        else 12 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f"{model_name} Not adapted for this experiment")))(),

        #model parameters
        "model_name" : model_name,
        "model_path" : f"./models/{model_name}/{model_name}-model",
        # "model_path" :
        "tokenizer_path" : f"./models/{model_name}/{model_name}-tokenizer",
        # "tokenizer_path" :
        "device": device,  

        #changable dataset parameters:
        "core_init_choice": "direct_init", # options: "direct_init", "init_and_decompose"
        "glue_type": glue_type, # glue, super_glue
        "dataset_name" : dataset_name, # glue for roberta are :mrpc, cola, sst2, qnli, super_glue for llama are: boolq, cb, copa, wsc
        
        #changeable hyperparameters
        "learning_rate": 1e-3
        # 1e-3 if "roberta-base" in model_name 
        # else 1e-5 if "llama-3-8b" in model_name
        # else 1e-5 if "llama-3-70b" in model_name
        # else ValueError(f"{model_name} Not adapted for this experiment"),
    }

    analysis =  train_without_ray(config)
    # df = pd.DataFrame(list(analysis.items()), columns=['metric', 'value'])
    # print(df)
    # filename = f"Expert-w/o-dd_{config["dataset_name"]}_{config["model_name"]}.csv"
    # df.to_csv(filename, index=False)
