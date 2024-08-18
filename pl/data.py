import lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
import logging

class MNISTDataModule(pl.LightningDataModule):
    #TODO: configure logging
    def __init__(self, n_teachers, batch_size, val_split, num_workers): 
        super().__init__()
        self.data_dir = '../data'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.n_teachers = n_teachers
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        
    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)
    
    def create_subsets(self, dataset):
        total_size = len(dataset)
        subset_size = total_size // self.n_teachers
        
        sizes = [subset_size] * self.n_teachers
        sizes[-1] += total_size - sum(sizes)
        
        subsets = random_split(dataset, sizes)
        
        for i, subset in enumerate(subsets):
            logging.info(f"Subset {i} : {len(subset)} samples")
        return subsets
    
    def setup(self):

        train_full = datasets.MNIST(self.data_dir, train=True)
        total_size = len(train_full)
        val_size = int(self.val_split * total_size)
        train_size = total_size  - val_size
        
        train_data, val_data = random_split(train_full, [train_size, val_size])
        
        self.train_subsets = self.create_subsets(train_data)
        self.val_subsets = self.create_subsets(val_data)
    
        self.student_train = Subset(datasets.MNIST(self.data_dir, train=False), list(range(9000)))
        self.student_test = Subset(datasets.MNIST(self.data_dir, train = False), list(range(9000, 10000)))
    
    
    def teacher_train_dataloaders(self):
        return [DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) for subset in self.train_subsets]

    def teacher_val_dataloaders(self):
        return [DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) for subset in self.val_subsets]
    
    def student_train_dataloader(self): #acts as 'test set' for teacher models
        return DataLoader(self.student_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) #TODO: check shuffle condition here
    
    def student_test_dataloader(self):
        return DataLoader(self.student_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        