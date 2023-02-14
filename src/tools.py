import numpy as np
import os
from pathlib import Path
import glob
import subprocess
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from matplotlib.ticker import AutoMinorLocator
from torch import tensor

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


def matplotlib_settings(small_size=12,medium_size=16,bigger_size=18):
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Palatino"
    })


def mse_snapshots(a,b,axis=1):
    return np.mean((a-b)**2,axis=axis)


def plot_1d_results(x, y, r, t=-1, diff=True):
    # data
    xs = x.shape[1]
    ys = y.shape[1]
    x = np.repeat(x,y.shape[1]//xs,axis=1)

    # plots
    fig, ax = plt.subplots(1,4 if diff else 3,figsize=(10,3),sharey=True,constrained_layout=True)
    vmin, vmax = np.min(y),np.max(y)
    im0 = ax[0].pcolormesh(x,cmap='inferno',vmin=vmin,vmax=vmax)    
    im1 = ax[1].pcolormesh(y,cmap='inferno',vmin=vmin,vmax=vmax)
    im2 = ax[2].pcolormesh(r,cmap='inferno',vmin=vmin,vmax=vmax)
    if diff:
        im3 = ax[3].pcolormesh(y-r,cmap='inferno',vmin=vmin,vmax=vmax)
    for axis,label in zip(ax,["$x$","$y$","$\hat{y}$","$\Delta$"]):
        axis.text(1-0.140,0.9,label,fontsize=14,c='black',transform=axis.transAxes)
        axis.text(1-0.135,0.9,label,fontsize=14,c='white',transform=axis.transAxes)
        circle = Circle((1-0.140+0.027,0.9+0.025), radius=0.05,
                            color='k', alpha=0.3, transform=axis.transAxes)
        axis.add_patch(circle)
    
    # labels
    for axis in ax:
        axis.set_xticks([0,ys])
    ax[0].set_xticklabels([0,xs])
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    # ax[0].set_ylabel('t (snapshot)')
    # ax[1].set_xlabel('x (px)')

    def _inset_plot(axis,p,y,ylim):
        axins = axis.inset_axes([0,t/x.shape[0] - 0.05,1, 0.1],transform=axis.transAxes)
        axins.plot(y,color='black')
        axins.plot(p,color='white')
        axins.set_facecolor('black')
        axins.patch.set_alpha(0.2)
        axins.set_xlim(0,ys)
        axins.set_ylim(ylim)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.spines[['left','right','top','bottom']].set_visible(False)
        return axins

    # inset plots
    if t >= 0:
        ylim = (np.min(y),np.max(y))
        for axis,p in zip(ax[:3],[x,y,r]):
            axins = _inset_plot(axis,p[t,:],y[t,:],ylim)
            # axis.hlines(100,xmin=0,xmax=ys,linestyles='-.',color='white')
        if diff:
            axins = _inset_plot(ax[3],y[t,:]-r[t,:],np.zeros(ys),ylim)
    return ax


def plot_2d_result(x,y,r,t=0,vmin=None,vmax=None,diff=True,apply_map=None,cbar=True):
    # data
    x,y,r = x[t,:,:], y[t,:,:], r[t,:,:]
    d = y - r
    if apply_map:
        x,y,r,d = map(apply_map,[x,y,r,d])

    # plots
    fig, ax = plt.subplots(1,4 if diff else 3,figsize=(11,2),constrained_layout=True)
    im0 = ax[0].pcolormesh(x)
    if (vmin is None) or (vmax is None):
        im1 = ax[1].pcolormesh(y)
        vmin, vmax = im1.get_clim()
    else:
        im1 = ax[1].pcolormesh(y,vmin=vmin,vmax=vmax)
    im2 = ax[2].pcolormesh(r,vmin=vmin,vmax=vmax)
    if diff:
        im3 = ax[3].pcolormesh(d,vmin=vmin,vmax=vmax)
    if cbar:        
        fig.colorbar(im3 if diff else im2, ax=ax.ravel().tolist(),
                    shrink=0.5 if diff else 0.68, pad=0.02)
    [axis.set_aspect('equal') for axis in ax];

    #labels
    for axis,label in zip(ax,["$x_i$","$y_i$","$\hat{y}_i$","$\Delta_i$"]):
        axis.text(1-0.140,0.75,label,fontsize=14,c='black',transform=axis.transAxes)
        axis.text(1-0.135,0.75,label,fontsize=14,c='white',transform=axis.transAxes)
        circle = Circle((1-0.140+0.027,0.9+0.025), radius=0.05,
                            color='k', alpha=0.3, transform=axis.transAxes)
    # for axis,label in zip(ax,['Input Frame','Desired Frame','Result','Diff']):
    #     axis.text(0.1,0.85,label,fontsize=11,c='white',transform=axis.transAxes)
    #     rect = patches.Rectangle((0.00,0.85),1.0,0.10,facecolor='black', alpha=0.3, transform=axis.transAxes)
    #     axis.add_patch(rect)

    ax[0].set_xticks([0,x.shape[1]])
    ax[0].set_yticks([0,x.shape[0]])
    ax[1].set_xticks([0,y.shape[1]])
    ax[1].set_yticks([0,y.shape[0]])
    ax[2].set_xticks([0,y.shape[1]])
    ax[2].set_yticks([])
    ax[3].set_xticks([0,y.shape[1]])
    ax[3].set_yticks([])

    for pos in ['top', 'bottom', 'right', 'left']:
        ax[2].spines[pos].set_linewidth(1.5)

    return (vmin,vmax), ax


def psd2(data, fftshift=True):
    """Compute 2D power spectrum for an n x m array 
    or a t x n x m time series of (n x m) arrays.
    """
    axes = [1,2] if data.ndim ==3 else [0,1]
    data_fft = np.fft.fftn(data,axes=axes)
    if fftshift:
        data_fft = np.fft.fftshift(data_fft,axes=axes)
    return np.abs(data_fft)**2


def timethis(func):
    """Creates a @timethis decorator. As long as timethis is in the namespace,
    you can put the @timethis decorator preceding any function to print its walltime.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper