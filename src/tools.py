# from datetime import datetime
import numpy as np
import os
from pathlib import Path
import glob
import subprocess
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle, ConnectionPatch

def dataset_split(dataset_x,dataset_y,valid_n):
    train_x = dataset_x[:-valid_n,:]
    train_y = dataset_y[:-valid_n,:]
    valid_x = dataset_x[-valid_n:,:]
    valid_y = dataset_y[-valid_n:,:]
    return train_x, train_y, valid_x, valid_y


def demean(a):
    return (a - a.mean(0))


def generate_video(plot_function,t_max,directory,framerate=20,dpi=300,
                   filename='out',rm_images=True,transparent=False):
    # generate images from plot_function
    if t_max >= 1e5:
        print("t_max too large!")
        return

    Path(directory).mkdir(exist_ok=True)
    for t in range(t_max):
        plot_function(t)
        plt.savefig(directory + "/%05d.png" % t, dpi=dpi, transparent=transparent)
        if t>0: plt.close()

    # create video via ffmpeg call
    previous_directory = os.getcwd()
    os.chdir(directory)
    try:
        filename_=f"{filename}.mp4"
        subprocess.call(f'ffmpeg -y -loglevel warning -framerate {framerate} -pattern_type glob -i *.png -c:v libx264 -pix_fmt yuv420p -vf crop=trunc(iw/2)*2:trunc(ih/2)*2 {filename_}'.split(' '))
        if rm_images: subprocess.call(['rm']+glob.glob("*.png"))
        print(f"created {filename_}")
    finally:
        os.chdir(previous_directory)


def mkdir_figs_vids():
    if not os.path.exists("figs"):
        os.mkdir("figs")
    if not os.path.exists("figs/vids"):
        os.mkdir("figs/vids")


def matplotlib_settings(small_size=10,medium_size=14,bigger_size=16):
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title


def plot_1d_results(x, r, y, loss=None):
    fig, ax = plt.subplots(1,3,figsize=(10,6), sharey=True)
    vmin, vmax = np.min(y),np.max(y)       
    im0 = ax[0].pcolormesh(x,cmap='inferno',vmin=vmin,vmax=vmax)    
    im1 = ax[1].pcolormesh(r,cmap='inferno',vmin=vmin,vmax=vmax)
    im2 = ax[2].pcolormesh(y,cmap='inferno',vmin=vmin,vmax=vmax)
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


def plot_2d_result(x, r, y, t=0, loss=None):
        fig, ax = plt.subplots(1,4,figsize=(10,2), sharey=False)
        t = int(t)
        im0 = ax[0].pcolormesh(x[t,:,:])
        im1 = ax[1].pcolormesh(r[t,:,:])
        vmin, vmax = im1.get_clim()
        s = r[t,:,:].shape
        im2 = ax[2].pcolormesh(y[t,:,:],vmin=vmin,vmax=vmax)
        im3 = ax[3].pcolormesh(y[t,:,:]-r[t,:,:],vmin=vmin,vmax=vmax)
        ax[0].set_ylabel('y (px)')
        ax[1].set_xlabel('x (px)')
        for axis,label in zip(ax,['Input Frame','Result','Desired Frame','Diff']):
            axis.text(0.1,0.90,label,fontsize=12,c='white',transform=axis.transAxes)
            rect = patches.Rectangle((0.00,0.90),1.0,0.10,facecolor='black', alpha=0.3, transform=axis.transAxes)
            axis.add_patch(rect)
        if loss is not None:
            ax[1].text(0.5,0.90,f"L2 = {np.round(loss,2)}",fontsize=10,c='w',transform=ax[1].transAxes)
        [axis.set_aspect('equal') for axis in ax];
        plt.tight_layout()
        plt.colorbar(im3,fraction=0.046*s[0]/s[1], pad=0.04)

