import numpy as np
from torch import optim, tensor
import torch.nn.functional as F
from ..torch_boilerplate import get_torchdevice, fit
from ..SDNN import SDNN, get_dataloader

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from pathlib import Path
from ..tools import *
matplotlib_settings()


DEVRUN = False
MAKEFIGS = True
MAKEVIDS = True

datafile = 'dh_dh_exp0_tr8_xr3_yr3_en1.npy'
with open(f"data/ael/{datafile}", 'rb') as f:
    dataset_x = np.load(f)
    dataset_y = np.load(f)
    shape_x = np.load(f)
    shape_y = np.load(f)

valid_n = 500
train_x, train_y, valid_x, valid_y = dataset_split(dataset_x,dataset_y,valid_n)

def demean(a):
    return (a - a.mean(0))

# def nonneg(a):
#     if a.min() < 0:
#         return a - a.min()

train_x,train_y = map(demean, [train_x,train_y])
valid_x,valid_y = map(demean, [valid_x,valid_y])

# train_x,train_y = map(nonneg, [train_x,train_y])
# valid_x,valid_y = map(nonneg, [valid_x,valid_y])

print("train_x","train_y",[x.shape for x in [train_x, train_y]])
print("valid_x","valid_y",[x.shape for x in [valid_x, valid_y]])


bs = 64
lr = 0.0025
epochs = 150
if DEVRUN:
    epochs = 3
l1_size = 8192
l2_size = 256
dropout_p = 0.3

dev = get_torchdevice()
train_dl = get_dataloader(train_x, train_y, bs,   shuffle=False)
valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=False )
model = SDNN(train_x.shape[-1], train_y.shape[-1], 
        l1_size=l1_size, l2_size=l2_size, dropout_p=dropout_p).to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_func = F.mse_loss
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


# retrieve model output
model.cpu()
train_out = model(tensor(train_x).float().unsqueeze(1)).detach().numpy().squeeze()
valid_out = model(tensor(valid_x).float().unsqueeze(1)).detach().numpy().squeeze()
train_loss = F.mse_loss(*map(tensor, [train_out, train_y])).numpy()
valid_loss = F.mse_loss(*map(tensor, [valid_out, valid_y])).numpy()
print(f"train_loss: {train_loss}")
print(f"valid_loss: {valid_loss}")


if MAKEFIGS:

    figbasename = Path(datafile).stem

    # unstack training set
    tr = 1 #TODO
    train_x_ = train_x[:,:train_x.shape[1]//tr]
    valid_x_ = valid_x[:,:valid_x.shape[1]//tr]

    def plot_ml_results(x, out, y, t=0, loss=None):
        fig, ax = plt.subplots(1,3,figsize=(10,2), sharey=False)
        # vmin = np.min(valid_y)
        # vmax = np.max(valid_y)
        t = int(t)
        im0 = ax[0].pcolormesh(x[t,:,:])
        im1 = ax[1].pcolormesh(out[t,:,:])
        im2 = ax[2].pcolormesh(y[t,:,:])
        ax[0].set_ylabel('y (px)')
        ax[1].set_xlabel('x (px)')
        for axis,label in zip(ax,['Input Frame','Result','Desired Frame']):
            axis.text(0.1,0.90,label,fontsize=12,c='white',transform=axis.transAxes)
            rect = patches.Rectangle((0.00,0.90),1.0,0.10,facecolor='black', alpha=0.3, transform=axis.transAxes)
            axis.add_patch(rect)
        if loss is not None:
            ax[1].text(0.5,0.90,f"L2 = {np.round(loss,2)}",fontsize=10,c='w',transform=ax[1].transAxes)
        plt.tight_layout()
        fig.colorbar(im1, ax=ax, pad=0.02)
        # ax[0].text(0.1,-0.1,"text", transform=axis.transAxes)
    
    plot_ml_results(train_x_.reshape(-1,*shape_x[1:]),
                    train_out.reshape(-1,*shape_y[1:]),
                    train_y.reshape(-1,*shape_y[1:]),
                    t=0, loss=train_loss)
    plt.savefig(f"figs/{figbasename}_train.png", transparent=False)

    plot_ml_results(valid_x_.reshape(-1,*shape_x[1:]),
                    valid_out.reshape(-1,*shape_y[1:]),
                    valid_y.reshape(-1,*shape_y[1:]),
                    t=0, loss=valid_loss)
    plt.savefig(f"figs/{figbasename}_valid.png", transparent=False)

    if MAKEVIDS:
        f = lambda t: plot_ml_results(train_x_.reshape(-1, *shape_x[1:]), 
                train_out.reshape(-1, *shape_y[1:]),
                train_y.reshape(-1, *shape_y[1:]),
                t=t, loss=train_loss)

        generate_video(f,50,'figs/mov/dh-dh',framerate=6,
                   filename=f"{figbasename}_train_{l1_size}_{l2_size}",rm_images=True,transparent=False)

        g = lambda t: plot_ml_results(valid_x_.reshape(-1, *shape_x[1:]), 
                valid_out.reshape(-1, *shape_y[1:]),
                valid_y.reshape(-1, *shape_y[1:]),
                t=t, loss=valid_loss)

        generate_video(g,50,'figs/mov/dh-dh',framerate=6,
                   filename=f"{figbasename}_valid_{l1_size}_{l2_size}",rm_images=True,transparent=False)