import numpy as np
import torch
from torch import nn,optim,tensor
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm

def get_torchdevice():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


dev = get_torchdevice()


def preprocess(x,y):
    dev = get_torchdevice()
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


def get_dataloader(x,y,bs,shuffle):    
    ds = TensorDataset(*map(tensor, (x,y)))
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return WrappedDataLoader(dl, preprocess)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class EarlyStopper:
    def __init__(self, patience=1, percent=0.01, verbose=True):
        self.patience = patience
        self.percent = percent
        self.verbose = True
        self.counter = 0
        self.min_valid_loss = np.inf

    def early_stop(self, valid_loss, epoch):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0
        elif valid_loss >= self.min_valid_loss * (1+self.percent):
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose: 
                    print(f"Early stop. Min validation loss {self.min_valid_loss} on epoch {epoch}")
                return True
        return False


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, es_patience=3, es_percent=0.01):
    
    def loss_batch(model, loss_func, xb, yb, opt=None):
        # loss = loss_func(model(xb), yb)
        loss = loss_func(model(xb).squeeze(), yb.squeeze()) #TODO
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), len(xb)
    
    early_stopper = EarlyStopper(patience=es_patience, percent=es_percent)

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        valid_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        pbar.set_description(f"epoch: {epoch} valid_loss: {valid_loss}")

        if early_stopper.early_stop(valid_loss, epoch):             
            break


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        xb = xb.view(-1, 784)
        xb = self.lin(xb)
        return xb
    
    
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.lamb = Lambda(lambda x: x.view(x.size(0), -1))

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.adaptive_avg_pool2d(xb, 1)
        return self.lamb(xb)
    

## alternate sequential form of Mnist_CNN            
# model = nn.Sequential(
#     nn.Conv2d(1,  16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#     nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#     nn.AdaptiveAvgPool2d(1),
#     Lambda(lambda x: x.view(x.size(0), -1)),
# ).to(dev)


## sample training code

# bs = 32
# lr = 0.1
# epochs = 4

# dev = get_torchdevice()
# train_dl = get_dataloader(train_x, train_y, bs,   shuffle=False)
# valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=True )
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# model = Mnist_CNN().to(dev)
# loss_func = F.cross_entropy
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)

