import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def create_dataloaders(train, batch_size):

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])


    loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "../data", 
            split=train,
            download= True,
            transform = data_transforms
            ),
        batch_size = batch_size, 
        shuffle = True
        )
    
    return loader



class NoisyDataset (Dataset):

    def __init__(self, dataloader, model, transform = None):

        self.dataloader = dataloader
        self.model = model
        self.transform = transform
        self.noisy_data = self.process_data()

    def split(self):
        pass

    def process_data(self):
        return [[data, torch.tensor(self.model(data))] for data, _ in self.dataloader]
    
    def __len__(self):
        return len(self.dataloader)
    
    def __getitem__(self, index):
        sample = self.noisy_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
