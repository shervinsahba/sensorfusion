import numpy as np
from ..torch_boilerplate import *
from ..SDNN import SDNN, get_dataloader

# import matplotlib.pyplot as plt
# from matplotlib import patches

DEVRUN = False
MAKEFIGS = True

datafile = 'ks_32_embed1.npy'
with open(f"data/ks/{datafile}", 'rb') as f:
    dataset_x = np.load(f)
    dataset_y = np.load(f)
    xn = np.load(f)

valid_n = 2000
train_x = dataset_x[:-valid_n,:]
train_y = dataset_y[:-valid_n,:]
valid_x = dataset_x[-valid_n:,:]
valid_y = dataset_y[-valid_n:,:]

def demean(a):
    return (a - a.mean(0))

train_x,train_y = map(demean, [train_x,train_y])
valid_x,valid_y = map(demean, [valid_x,valid_y])

print("train_x","train_y",[x.shape for x in [train_x, train_y]])
print("valid_x","valid_y",[x.shape for x in [valid_x, valid_y]])


bs = 128
lr = 0.001
epochs = 80
if DEVRUN:
    epochs = 3

dev = get_torchdevice()
train_dl = get_dataloader(train_x, train_y, bs,   shuffle=False)
valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=True )
model = SDNN(train_x.shape[-1], train_y.shape[-1], l1_size=(l1_size:=8192), l2_size=(l2_size:=64)).to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_func = F.mse_loss
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


model.cpu()
train_out = model(tensor(train_x).float().view(-1,1,train_x.shape[-1])).detach().numpy().squeeze()
valid_out = model(tensor(valid_x).float().view(-1,1,valid_x.shape[-1])).detach().numpy().squeeze()
train_loss = F.mse_loss(*map(tensor, [train_out, train_y])).numpy()
valid_loss = F.mse_loss(*map(tensor, [valid_out, valid_y])).numpy()
print(f"train_loss: {train_loss}")
print(f"valid_loss: {valid_loss}")


if MAKEFIGS:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import os
    if not os.path.exists("figs"):
        os.mkdir("figs")

    def plot_ml_results(x, out, y, loss=None):
        fig, ax = plt.subplots(1,3,figsize=(10,6), sharey=True)
        vmin = np.min(valid_y)
        vmax = np.max(valid_y)
        im0 = ax[0].pcolormesh(x,   cmap='inferno', vmin=vmin, vmax=vmax)
        im1 = ax[1].pcolormesh(out, cmap='inferno', vmin=vmin, vmax=vmax)
        im2 = ax[2].pcolormesh(y,   cmap='inferno', vmin=vmin, vmax=vmax)
        ax[0].set_ylabel('t (snapshot)')
        ax[1].set_xlabel('x (px)')
        for axis,label in zip(ax,['Input','Result','Desired Output']):
            axis.text(0.1,0.95,label,fontsize=16,c='white',transform=axis.transAxes)
            rect = patches.Rectangle((0.00,0.935),1.0,0.05,facecolor='black', alpha=0.3, transform=axis.transAxes)
            axis.add_patch(rect)
        if loss is not None:
            ax[1].text(0.5,0.95,f"L2 = {np.round(loss)}",fontsize=12,c='w',transform=ax[1].transAxes)
        plt.tight_layout()
        fig.colorbar(im1, ax=ax, pad=0.02)

    plot_ml_results(train_x[:valid_n,:xn], train_out[:valid_n,:], train_y[:valid_n,:], train_loss)
    plt.savefig(f"figs/ks-{xn}-train"+".png", transparent=False)
        
    plot_ml_results(valid_x[:,:xn], valid_out, valid_y, valid_loss)
    plt.savefig(f"figs/ks-{xn}-valid"+".png", transparent=False)
