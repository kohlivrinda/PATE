import torch
from teacher import Teacher
from model import Classifier
from data import create_dataloaders, NoisyDataset
from utils import accuracy, loop
from student import Student

class Arguements():

    def __init__(self):
        self.batchsize = 64
        self.test_batchsize = 10
        self.epochs=50
        self.student_epochs=10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.n_teachers=50
        self.save_model = False

args = Arguements()

train_loader = create_dataloaders(True, args.batchsize)
test_loader = create_dataloaders(False, args.test_batchsize)

teacher = Teacher(args, Model, n_teachers=args.n_teachers)
teacher.train(train_loader)

targets=[]
predict=[]
counts = []

for data,target in test_loader:
    
    targets.append(target)
    predict.append(teacher.predict(data)["predictions"])
    counts.append(teacher.predict(data)["model_counts"])
    
print("Accuracy: ",accuracy(torch.tensor(predict),targets))

print("\nTraining Student\n\n")

student=Student(args,Model())
N = NoisyDataset(train_loader,teacher.predict)
student.train(N)

results=[]
targets=[]

total=0.0
correct=0.0

for data,target in test_loader:
    
    predict_lol=student.predict(data)
    correct += float((predict_lol == (target)).sum().item())
    total+=float(target.size(0))
    
print(" Pvt baseline: " , (correct/total)*100)

counts_lol = torch.stack(counts).contiguous().view(50, 10000)
predict_lol = torch.tensor(predict).view(10000)

data_dep_eps, data_ind_eps = teacher.analyze(counts_lol, predict_lol, moments=20)
print("Epsilon: ", teacher.analyze(counts_lol, predict_lol))