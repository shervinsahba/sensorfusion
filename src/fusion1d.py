import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import optim, tensor
import torch.nn.functional as F
from .torch_boilerplate import get_torchdevice, fit
from .SDNN import *
from .tools import *

matplotlib_settings()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', help='data.npy')
    parser.add_argument('--valid_n', help='n validation snapshots', nargs='?', default=2000, type=int)    
    parser.add_argument('--epochs', help='training steps', nargs='?', default=80, type=int)
    parser.add_argument('--bs', help='batch size', nargs='?', default=128, type=int)
    parser.add_argument('--lr', help='learning rate', nargs='?', default=0.001, type=float)
    parser.add_argument('--l1', help='layer 1 neurons', nargs='?', default=8192, type=int)
    parser.add_argument('--l2', help='layer 2 neurons', nargs='?', default=256, type=int)
    parser.add_argument('--dp', help='dropout percentage', nargs='?', default=0.3, type=float)
    parser.add_argument('--DEVRUN', help='dev mode quickrun', nargs='?', default=False, type=bool)
    parser.add_argument('--MAKEFIGS', help='toggle making figures on', nargs='?', default=True, type=bool)
    parser.add_argument('--MAKEVIDS', help='toggle making movies on', nargs='?', default=True, type=bool)
    return parser.parse_args()


def main(datafile,valid_n,epochs,bs,lr,l1,l2,dp,
        DEVRUN=False,MAKEFIGS=False,MAKEVIDS=False):

    ## load data
    if datafile is None:
        raise TypeError("Provide a datafile from data/ks/, e.g. data.npy")
    with open(f"data/ks/{datafile}", 'rb') as f:
        dataset_x = np.load(f)
        dataset_y = np.load(f)
        xn = np.load(f)

    ## handle training data
    train_x, train_y, valid_x, valid_y = dataset_split(dataset_x,dataset_y,valid_n)
    train_x,train_y,valid_x,valid_y = map(demean, [train_x,train_y,valid_x,valid_y])
    print("train_x","train_y",[x.shape for x in [train_x, train_y]])
    print("valid_x","valid_y",[x.shape for x in [valid_x, valid_y]])

    ## ml
    dev = get_torchdevice()
    train_dl = get_dataloader(train_x, train_y, bs,   shuffle=False)
    valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=True )
    model = SDNN(train_x.shape[-1], train_y.shape[-1], l1=l1, l2=l2, dp=dp).to(dev)
    # opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss
    if DEVRUN: 
        epochs = 3
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)

    # retrieve model output
    model.cpu()
    train_r, train_loss = predict(train_x, train_y, model)
    valid_r, valid_loss = predict(valid_x, valid_y, model)
    print(f"train_loss: {train_loss}")
    print(f"valid_loss: {valid_loss}")

    ## setup for figs or vids
    figbasename = Path(datafile).stem
    mkdir_figs_vids()

    if MAKEFIGS:  
        pn = max(500,valid_n)  # number of snapshots to plot
        t = pn * 3//5          # pick a snapshot to plot about 0.6 up the plot, aesthetically speaking
        plot_1d_results(train_x[:pn,:xn], train_y[:pn,:], train_r[:pn,:], t=t)
        plt.savefig(f"figs/{figbasename}_train_{l1}_{l2}.png", transparent=False)
            
        plot_1d_results(valid_x[:pn,:xn], valid_y[:pn,:], valid_r[:pn,:], t=t)
        plt.savefig(f"figs/{figbasename}_valid_{l1}_{l2}.png", transparent=False)

        if MAKEVIDS:
            # TODO. There are no vids yet.
            pass

    return (train_x, train_y, train_r), (valid_x, valid_y, valid_r)


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))
