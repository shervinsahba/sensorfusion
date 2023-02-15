import numpy as np
from torch import nn,tensor
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from .torch_boilerplate import *

class SDNN(nn.Module):
    def __init__(self,input_size,output_size,l1=1024,l2=128,d1=0.20,d2=0.00):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        
        self.learn_features = nn.Sequential(         
            nn.Linear(input_size, self.l1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1),
            )
        
        self.learn_coef = nn.Sequential(            
            nn.Linear(self.l1, self.l2),
            nn.ReLU(inplace=True),  
            nn.BatchNorm1d(1),  
            )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(self.l2, self.output_size),
            )
        
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
        xb = self.learn_features(xb)
        xb = F.dropout(xb, p=self.d1, training=self.training)
        xb = self.learn_coef(xb)
        xb = F.dropout(xb, p=self.d2, training=self.training)
        xb = self.learn_dictionary(xb) 
        return xb


def preprocess(x,y):
    dev = get_torchdevice()
    return x.unsqueeze(1).float().to(dev), \
           y.unsqueeze(1).float().to(dev)


def get_dataloader(x,y,bs,shuffle):    
    ds = TensorDataset(*map(tensor, (x,y)))
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return WrappedDataLoader(dl, preprocess)


def predict(x, y, model):
    r = model(tensor(x).float().unsqueeze(1)).detach().numpy().squeeze()
    mse_loss = np.mean((r-y)**2)
    return r, mse_loss

