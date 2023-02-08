from torch import nn,tensor
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from .torch_boilerplate import *

class SDNN(nn.Module):
    def __init__(self,input_size,output_size,
                 l1_size=1024,l2_size=128,dropout_p=0.5):
        super().__init__()
        self.input_size = input_size
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        
        self.learn_features = nn.Sequential(         
            nn.Linear(input_size, self.l1_size),
            nn.ReLU(inplace=True),
            # nn.PReLU(num_parameters=input_size), 
            nn.BatchNorm1d(1),  
            )
        
        self.learn_coef = nn.Sequential(            
            nn.Linear(self.l1_size, self.l2_size),
            nn.ReLU(inplace=True),  
            # nn.PReLU(num_parameters=l1_size),
            nn.BatchNorm1d(1),  
            )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(self.l2_size, self.output_size),
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
        xb = F.dropout(xb, p=self.dropout_p, training=self.training)
        xb = self.learn_coef(xb)
        xb = self.learn_dictionary(xb) 
        return xb


def preprocess(x,y):
    dev = get_torchdevice()
    return x.unsqueeze(1).float().to(dev), \
           y.unsqueeze(1).float().to(dev), \


def get_dataloader(x,y,bs,shuffle):    
    ds = TensorDataset(*map(tensor, (x,y)))
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return WrappedDataLoader(dl, preprocess)