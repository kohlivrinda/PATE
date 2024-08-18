import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import loop

class Student:

    def __init__(self, args, model):
        self.args = args
        self.model = model

    def predict(self, data):
        return torch.max(self.model(data),1)[1]
    
    def train(self, dataset):
        for epoch in range(0, self.args.student_epochs):
            loop(dataset, epoch, self.model)

    def save_model(self):
        torch.save(self.model.state_dict(), "models/" + "student_model")
        