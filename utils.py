import torch

def accuracy(preds, dataset):
    total , correct = 0.0, 0.0
    
    correct = sum((p.long() == d.long()).sum().item() for p,d in zip(preds, dataset))
    total = sum(len(d) for d in dataset)
    return ((correct/total)*100)
