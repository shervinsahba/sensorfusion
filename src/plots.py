import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors
import os
from pathlib import Path
import glob
import subprocess
import gc
from tqdm.auto import tqdm
from .tools import radial_psd, psd2, timethis


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
        # "font.family": "Palatino"  # TODO font family may cause a memory leak in plot loops
    })



def mkdir_figs_vids():
    if not os.path.exists("figs"):
        os.mkdir("figs")
    if not os.path.exists("figs/vids"):
        os.mkdir("figs/vids")


def normalize_colormaps(images):
    """Takes an array of matplotlib image handles and normalizes 
    the images' color maps. Useful for normalizing subplots.
    """
    vmin = min(im.get_array().min() for im in images)
    vmax = max(im.get_array().max() for im in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    return vmin,vmax


@timethis
def generate_video(plot_function,t_max,start=0,directory='figs',filename='out',
                   framerate=30,dpi=300,rm_images=True,transparent=False):
    if t_max >= 1e5:
        raise ValueError(f"Do you really want to save {t_max} images? Try smaller.")

    Path(directory).mkdir(exist_ok=True)  # create figure dir if doesn't exist
    tmp = str(plot_function.__hash__())   # tmp identifier so you don't clobber other files
    
    matplotlib.use('agg')
    print("Reseting matplotlib font family to default serif to avoid a memory leak.")
    plt.rcParams.update({"font.family": "serif"})
    # generate images from plot_function
    # TODO note that matplotlib has memory leaks. see for example:
    # https://github.com/matplotlib/matplotlib/issues/20300
    # https://github.com/matplotlib/matplotlib/issues/25406
    # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.6.0.html#garbage-collection-is-no-longer-run-on-figure-close
    # These will cause this loop to get real big, real fast 
    # perhaps regardless of plt.close and gc.collect
    for t in tqdm(range(start,t_max)):  # generate images from plot_function
        plot_function(t)
        plt.gcf().savefig(f"{Path(directory,tmp)}_{t:05d}.png", dpi=dpi, transparent=transparent)
        plt.close()
        gc.collect()
 
    # call ffmpeg
    filename=f"{Path(directory,filename)}.mp4"
    cmdstring = ('ffmpeg', '-loglevel', 'warning', '-y', '-r', f'{framerate}',
        '-pattern_type', 'glob', '-i', f'{Path(directory,tmp)}*.png',  # input images are globbed pngs
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',  # video codec and pixel format
        '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2', # truncate to an even number of pixels (needed for some codecs)
        filename)
    try:
        subprocess.call(cmdstring)
        print(f"created {filename}")
    except Exception as e:
        print("video generation failed:", e)

    if rm_images:  # remove temporary images
        subprocess.call(['rm']+glob.glob(f"{Path(directory,tmp)}*.png"))


def plot_1d_results(x, y, r, t=None, diff=False, cmap='inferno', fig=None, ax=None):
    xs = x.shape[1]
    ys = y.shape[1]
    data = [np.repeat(x,ys//xs,axis=1), y, r]
    labels = ["$X$","$Y$","$\hat{Y}$"]
    if diff: 
        data.append(y-r)
        labels.append("$\Delta$")
    
    fig, ax = plt.subplots(1,len(data),figsize=(10,3),sharey=True,constrained_layout=True)
    images = [axis.pcolormesh(p, cmap=cmap) for axis,p in zip(ax,data)]
    normalize_colormaps(images)

    for axis,label in zip(ax,labels):
        axis.text(0.87,0.9,label,fontsize=18,c='w',transform=axis.transAxes)
        axis.add_patch(Rectangle((0.85,0.88),0.125,0.12,color='k',alpha=0.85,transform=axis.transAxes))
        axis.set_xticks([0,ys])
    ax[0].set_xticklabels([0,xs])
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())

    def _inset_style(axins):
        axins.set_facecolor('black')
        axins.patch.set_alpha(0.4)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.spines[['left','right','top','bottom']].set_visible(False)

    def _inset_plot(axis,p,ylim):
        axins = axis.inset_axes([0,t/x.shape[0] - 0.05,1, 0.1],transform=axis.transAxes)
        axins.plot(p,color='white',lw=2.5)
        axins.set_xlim(0,ys)
        axins.set_ylim(ylim)
        _inset_style(axins)
        return axins

    def _inset_label(axis,text):
        axins = axis.inset_axes([0.85,t/x.shape[0] - 0.14,0.125,0.09],transform=axis.transAxes)
        axins.text(0.13,0.32,text,fontsize=18,color='w',transform=axins.transAxes)
        axins.set_facecolor('black')
        _inset_style(axins)        
        return axins

    if t >= 0:      # if a time a specified, make inset plots
        labels_inset = ["$x_i$","$y_i$","$\hat{y}_i$"]
        if diff: labels_inset.append("$\delta_i$")
        
        for axis,p,label in zip(ax,data,labels_inset):
            _inset_plot(axis,p[t,:],(np.min(y),np.max(y)))
            _inset_label(axis,label)
    
    return fig, ax


def plot_2d_result(x, y, r, t=0, vmin=None, vmax=None, diff=False, apply_map=None, cbar=True):
    data = [x[t,:,:], y[t,:,:], r[t,:,:]]
    labels = ["$x_i$","$y_i$","$\hat{y}_i$"]
    if diff:
        data.append(y[t,:,:] - r[t,:,:])
        labels.append("$\delta_i$")
    if apply_map:
        data = [apply_map(z) for z in data]

    fig, ax = plt.subplots(1,len(data),figsize=(11,2),constrained_layout=True)
    images = [axis.pcolormesh(p,vmin=vmin,vmax=vmax) for axis,p in zip(ax,data)]
    if None in [vmin,vmax]:
        vmin,vmax = normalize_colormaps(images)
    
    if cbar:        
        fig.colorbar(images[-1], ax=ax.ravel().tolist(),
            shrink=0.5 if diff else 0.68, pad=0.02)

    for axis,label in zip(ax,labels):
        axis.set_aspect('equal')
        axis.text(0.87,0.82,label,fontsize=18,c='w',transform=axis.transAxes)
        patch = Rectangle((0.85,0.75),0.125,0.25,color='k',alpha=0.85,transform=axis.transAxes)
        axis.add_patch(patch)

    ax[0].set_xticks([0,data[0].shape[1]])
    ax[0].set_yticks([0,data[0].shape[0]])
    ax[1].set_xticks([0,data[1].shape[1]])
    ax[1].set_yticks([0,data[1].shape[0]])
    for axis in ax[2:]:
        axis.set_xticks([0,data[1].shape[1]])
        axis.set_yticks([])

    return fig, ax, (vmin,vmax)


def plot_psd(data):
    # subplot grid setup
    fig,ax=plt.subplots(3,3,figsize=(10.5,5))
    gs = ax[0,0].get_gridspec()
    for axis in ax[0:,:2].flatten():
        axis.remove()
    axbig = fig.add_subplot(gs[0:,:2])
    
    # radial power spectra
    rpsds = [radial_psd(z) for z in data]
    rpsds[0] = np.repeat(rpsds[0], rpsds[1].shape[1]//rpsds[0].shape[1], axis=1)

    # plotting setup
    labels = ['$x_i$', '$y_i$', '$\hat{y}_i$']
    colors = ['tab:blue', 'tab:orange', 'tab:purple']
    rlim = data[1].shape[1] // 2

    for p,c,label in zip(rpsds,colors,labels):
        xlim = p.shape[1]
        axbig.fill_between(range(xlim),p.min(axis=0),p.max(axis=0),color=c,alpha=0.2)
        axbig.plot(p.mean(axis=0),':',color=c,lw=2,ms=4)
        axbig.plot(p.mean(axis=0)[:rlim],color=c,lw=2,ms=4,label=label)    
    axbig.set_xlim([0,xlim])
    axbig.set_ylim([min(np.min(p) for p in rpsds), max(np.max(p) for p in rpsds)])
    axbig.set_xlabel('r',labelpad=-10)
    axbig.set_ylabel('$\overline{\log \mathrm{PSD}_r}$', labelpad=-5)
    axbig.legend()

    labels_psd = [f"$\overline{{\mathrm{{PSD}}({l[1:-1]})}}$" for l in labels]
    images = [axis.pcolormesh(psd2(p).mean(axis=0)) for axis,p in zip(ax[:,2],data)]
    normalize_colormaps(images)

    for axis,p,c,label in zip(ax[:,2],data,colors,labels_psd):       
        axis.set_aspect('equal')
        axis.set_xticks([0,p[0,:,:].shape[1]])
        axis.set_yticks([0,p[0,:,:].shape[0]])
        axis.tick_params(axis=u'both', which=u'both',length=0)
        axis.text(0.7,0.8,label,fontsize=13,c='w',transform=axis.transAxes)
        for loc in ['top','bottom','left','right']:
            axis.spines[loc].set_color(c) 
            axis.spines[loc].set_linestyle('dotted') 
        axis.add_patch(Circle((p.shape[2]/2+0.5,p.shape[1]/2+0.5),p.shape[1]/2+0.5,ec=c,fill=False,lw=1,ls='-'))
        
    return fig, ax

