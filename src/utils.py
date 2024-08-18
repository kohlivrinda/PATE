import torch
import torch.optim as optim
import torch.functional as F

def accuracy(preds, dataset):
    total , correct = 0.0, 0.0
    
    correct = sum((p.long() == d.long()).sum().item() for p,d in zip(preds, dataset))
    total = sum(len(d) for d in dataset)
    return ((correct/total)*100)

def loop(self, dataset, epoch, model):

    optimizer = optim.SGD(self.model.parameters(),
                            lr = self.args.lr,
                            momentum = self.args.momentum)
    iters = 0
    loss = 0.0

    for (data, target) in dataset:
        optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        iters+=1

    print(f"\n Epoch: {epoch} \n Loss: {loss.item()}")