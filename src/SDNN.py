import numpy as np
from torch import nn, tensor
from torch.utils.data import TensorDataset, DataLoader
from .torch_boilerplate import *

class SDNN(nn.Module):
    def __init__(self,input_size,output_size,l1=1024,l2=128,dp=0.20):
        super().__init__()
        
        self.layer1 = nn.Sequential(         
            nn.Linear(input_size, l1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1))
        self.layer2 = nn.Sequential(      
            nn.Dropout(dp),      
          nn.Linear(l1, l2),
            nn.ReLU(inplace=True),  
            nn.BatchNorm1d(1))
        self.layer3 = nn.Linear(l2, output_size)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, xb):
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        return xb


def preprocess(x, y):
    dev = get_torchdevice()
    return x.unsqueeze(1).float().to(dev), \
           y.unsqueeze(1).float().to(dev)


def get_dataloader(x, y, bs, shuffle):    
    ds = TensorDataset(*map(tensor, (x,y)))
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return WrappedDataLoader(dl, preprocess)


def predict(x, y, model):
    """Make predictions using the given model and calculate the mean squared error loss."""
    r = model(tensor(x).float().unsqueeze(1)).squeeze().detach().numpy()
    mse_loss = np.mean((r-y)**2)
    return r, mse_loss


def train_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
                bs: int, lr: float, l1: int, l2: int, dp: float, epochs: int, devrun: bool) -> tuple:
    """Train the model."""
    dev = get_torchdevice()
    train_dl = get_dataloader(train_x, train_y, bs, shuffle=False)
    valid_dl = get_dataloader(valid_x, valid_y, bs * 2, shuffle=True)
    model = SDNN(train_x.shape[-1], train_y.shape[-1], l1=l1, l2=l2, dp=dp).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss
    if devrun:
        epochs = 3
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    return model


def evaluate_model(model, train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray) -> tuple:
    """Evaluate the model and get predictions."""
    model.cpu()  # retrieve model to cpu
    train_r, train_loss = predict(train_x, train_y, model)
    valid_r, valid_loss = predict(valid_x, valid_y, model)
    return train_r, train_loss, valid_r, valid_loss