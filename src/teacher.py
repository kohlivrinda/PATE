import torch
from torch.distributions.laplace import Laplace
from utils import loop


class Teacher:

    def __init__(self, args, model, n_teachers = 1, epsilon = 0.5):
        self.n_teachers = n_teachers
        self.model = model
        self.args = args
        self.epsilon = epsilon
        self.models = self.init_models()

    def init_models(self):
        return {f"model_{index}": self.model() for index in range(self.n_teachers)}

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
                loop(splits[index], 1, self.models[model_name])


    def batch(self, votes, batch_size):
        counts = []
        for vote_batch in votes:
            batch_counts = torch.zeros((len(vote_batch), 10))

            for idx, vote in enumerate(vote_batch):
                batch_counts[idx][vote]+=1

            counts.append(batch_counts)

        return counts
        
    def aggregate(self, model_votes, batch_size):
        counts = torch.zeros((batch_size, 10))
        for model in model_votes:
            for tensor in model_votes[model]:
                for val in tensor:
                    counts[val] += 1

        return counts

    def save_models(self):
        for _, (model_name, model) in enumerate(self.models.items()):
            torch.save(model.state_dict(), f"models/{model_name}")

        print("\n MODELS SAVED \n")

    def load_models(self):
        path_name = "model_"

        for i in range(self.args.n_teachers):
            modelA = self.model()
            modelA.load_state_dict(torch.load(f"models/{path_name}{i}"))
            self.models[f"{path_name}{i}"] = modelA

    def predict(self, data):

        model_preds = {}
        for model_name, model in self.models.items():
            output = model(data).max(dim=1)[1]
            model_preds[model_name] = [output]

        counts = self.add_noise(self.aggregate(model_preds, len(data)))
        predictions = [torch.tensor(batch.max(dim=0)[1].long()).clone().detach() for batch in counts]

        return predictions
