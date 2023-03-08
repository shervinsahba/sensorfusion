import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import optim, tensor
import torch.nn.functional as F
from .torch_boilerplate import get_torchdevice, fit
from .SDNN import *
from .tools import *
from .plots import *

matplotlib_settings()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', help='data.npy')
    parser.add_argument('--valid_n', help='n validation snapshots', nargs='?', default=500, type=int)    
    parser.add_argument('--epochs', help='training steps', nargs='?', default=2000, type=int)
    parser.add_argument('--bs', help='batch size', nargs='?', default=512, type=int)
    parser.add_argument('--lr', help='learning rate', nargs='?', default=1e-4, type=float)
    parser.add_argument('--l1', help='layer 1 neurons', nargs='?', default=8192, type=int)
    parser.add_argument('--l2', help='layer 2 neurons', nargs='?', default=256, type=int)
    parser.add_argument('--dp', help='dropout percentage', nargs='?', default=0.3, type=float)
    parser.add_argument('--DEVRUN', help='dev mode quickrun', nargs='?', default=False, type=bool)
    parser.add_argument('--MAKEFIGS', help='toggle making figures on', nargs='?', default=True, type=bool)
    parser.add_argument('--MAKEVIDS', help='toggle making movies on', nargs='?', default=True, type=bool)
    return parser.parse_args()


@timethis
def main(datafile,valid_n,epochs,bs,lr,l1,l2,dp,
        DEVRUN=False,MAKEFIGS=False,MAKEVIDS=False):

    ## load data
    if datafile is None:
        raise TypeError("Provide a datafile from data/../, e.g. data.npy")
    with open(f"data/ael/{datafile}", 'rb') as f:
        dataset_x = np.load(f)
        dataset_y = np.load(f)
        shape_x = np.load(f)
        shape_y = np.load(f)

    ## handle training data
    train_x,train_y,valid_x,valid_y = dataset_split(dataset_x,dataset_y,valid_n)
    train_x,train_y,valid_x,valid_y = map(demean, [train_x,train_y,valid_x,valid_y])
    print("train_x","train_y",[x.shape for x in [train_x, train_y]])
    print("valid_x","valid_y",[x.shape for x in [valid_x, valid_y]])

    ## ml
    dev = get_torchdevice()
    train_dl = get_dataloader(train_x, train_y, bs,   shuffle=False)
    valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=True )
    model = SDNN(train_x.shape[-1], train_y.shape[-1], l1=l1, l2=l2, dp=dp).to(dev)
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

    # unstack training set
    figbasename = Path(datafile).stem
    en = int(figbasename.split("en")[1].split("_")[0])
    sensors = figbasename.split("_")[:2]
    train_x = train_x[:,:train_x.shape[1]//en]
    valid_x = valid_x[:,:valid_x.shape[1]//en]

    #reshape data
    data_train = [
        train_x.reshape(-1,*shape_x[1:]),
        train_y.reshape(-1,*shape_y[1:]),
        train_r.reshape(-1,*shape_y[1:])
    ]

    data_valid = [
        valid_x.reshape(-1,*shape_x[1:]),
        valid_y.reshape(-1,*shape_y[1:]),
        valid_r.reshape(-1,*shape_y[1:])
    ]

    if MAKEFIGS:
        mkdir_figs_vids()

        _,_,(vmin,vmax) = plot_2d_result(*data_train)
        plt.savefig(f"figs/{figbasename}_train_{l1}_{l2}.png",
            transparent=False, bbox_inches='tight', dpi=300)

        plot_2d_result(*data_valid)
        plt.savefig(f"figs/{figbasename}_valid_{l1}_{l2}.png",
            transparent=False, bbox_inches='tight', dpi=300)

        plot_psd(data_train)
        plt.savefig(f"figs/{figbasename}_valid_psds_{l1}_{l2}.png",
            transparent=False, bbox_inches='tight', dpi=300)

        plot_psd(data_valid)
        plt.savefig(f"figs/{figbasename}_valid_psds_{l1}_{l2}.png",
            transparent=False, bbox_inches='tight', dpi=300)

        if MAKEVIDS:
            f = lambda t: plot_2d_result(*data_train,t=t,vmin=vmin,vmax=vmax)
            g = lambda t: plot_2d_result(*data_valid,t=t,vmin=vmin,vmax=vmax)
            generate_video(f,50,'figs/vids/',framerate=6,filename=f"{figbasename}_train_{l1}_{l2}")
            generate_video(g,50,'figs/vids/',framerate=6,filename=f"{figbasename}_valid_{l1}_{l2}")

    return data_train, data_valid


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))

