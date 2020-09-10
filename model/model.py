import tensorflow
import transformers
import torch
import pandas as pd
import numpy as np
import gc

from collections import defaultdict
from transformers import (
    AutoModelWithLMHead, AutoTokenizer, 
    get_linear_schedule_with_warmup,
    Trainer, TrainingArguments
)
from torch import nn, optim
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



MAX_LEN = 200
BATCH_SIZE = 1
EPOCHS = 5

train_data = pd.read_json(r'/home/scott/projects/reflector/SamSum Corpus/train.json')
val_data = pd.read_json(r'/home/scott/projects/reflector/SamSum Corpus/val.json')
test_data = pd.read_json(r'/home/scott/projects/reflector/SamSum Corpus/test.json')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", truncation = True)
tokenizer.eos_token = "<eos>"
tokenizer.sep_token = "<sep>"
tokenizer.bos_token = "<start>"
tokenizer.pad_token = "<pad>"

train_encodings = tokenizer(list(train_data.dialogue), 
    padding = True, truncation = True, max_length = MAX_LEN)
val_encodings = tokenizer(list(val_data.dialogue), 
    padding = True, truncation = True, max_length = MAX_LEN)
test_encodings = tokenizer(list(test_data.dialogue), 
    padding = True, truncation = True, max_length = MAX_LEN)

train_labels = tokenizer( list(train_data.summary), 
    padding = True, truncation = True, max_length = MAX_LEN)
val_labels = tokenizer( list(val_data.summary), 
    padding = True, truncation = True, max_length = MAX_LEN)
test_labels = tokenizer( list(test_data.summary), 
    padding = True, truncation = True, max_length = MAX_LEN)

class SamSumDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.context_len = context_len
        self.summary_len = summ_len
        self.summary = self.data.summary
        self.context = self.data.context

    def __len__(self):
        return len(self.summary)

    def __getitem__(self, index):
        context = str(self.context[index])
        #context = ' '.join(context.split())

        summary = str(self.summary[index])
        #summary = ' '.join(summary.split())

        source = self.tokenizer.batch_encode_plus(
            [context], max_length = self.context_len, 
            pad_to_max_length = True,return_tensors='pt'
            )
        target = self.tokenizer.batch_encode_plus(
            [summary], max_length= self.summary_len, 
            pad_to_max_length = True,return_tensors='pt'
            )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


# Model Training Setup
device = torch.device("cuda") 
total_steps = len(train_data)*EPOCHS
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
model.train()
#optimizer = Adagrad(model.parameters(), lr=5e-2)
model.to(device) 
# Set to half precision
# model.half() no good?
#loss_fn = nn.CrossEntropyLoss().to(device)


# DATA SETUP
train_dataset = SamSumDataset(
    encodings = train_encodings,
    labels = train_labels
    )

val_dataset = SamSumDataset(
    encodings = val_encodings,
    labels = val_labels
    )

# Data Loaders

train_data_loader = DataLoader(
    dataset = train_dataset,
    shuffle = True,
    num_workers = 10,
    pin_memory = True,
    batch_size = BATCH_SIZE
    )

val_data_loader = DataLoader(
    dataset = val_dataset,
    shuffle = True,
    num_workers = 10,
    pin_memory = True,
    batch_size = BATCH_SIZE
    )



# Train

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, 
            decoder_input_ids = y_ids, lm_labels = lm_labels)
        loss = outputs[0]
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()