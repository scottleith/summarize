import torch
import transformers
import numpy as np

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """ docstring? """
    losses = []
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        targets = d['targets'].to(device)
        outputs = model(input_ids = input_ids, labels = targets)
        loss = loss_fn(outputs, targets)
        losses.append( loss.item() )
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    losses = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            targets = d['targets'].to(device)
            outputs = model(input_ids = input_ids, labels = targets)
            loss = loss_fn(outputs, targets)
            losses.append( loss.item() )
    return np.mean(losses)

