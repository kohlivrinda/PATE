import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.laplace import Laplace
from utils import accuracy


class Teacher:

    def __init__(self, args, model, n_teachers = 1, epsilon = 0.5):
        self.n_teachers = n_teachers
        self.model = model
        self.args = args
        self.init_models()
        self.epsilon = epsilon
        self.models = {}

    def init_models(self):
        self.models = {f"model_{index}": self.model() for index in range(self.n_teachers)}

    def addnoise(self, x):
        m = Laplace(torch.tensor([0.0]), torch.tensor([self.epsilon]))
        count = x + m.sample()
        return count
    
    def split (self, dataset):

        ratio = len(dataset)//self.n_teachers
        split = [[] for _ in range(self.n_teachers)]
        iters = 0

        for data, target in dataset:
            split[iters // ratio].append([data, target])
            iters += 1

        return split
    

    
    def train(self, dataset):

        splits = self.split(dataset)
        for epoch in range(1, self.args.epochs+1):
            for index, model_name in enumerate(self.models):
                print(f"TRAINING {model_name}")
                print(f"EPOCH: {epoch}")
                self.loop_body(splits[index], model_name, 1)


    def loop_body (self, split, model_name, epoch):
        model = self.models[model_name]
        optimizer = optim.SGD
