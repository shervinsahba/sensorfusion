from datetime import datetime
import os
from pathlib import Path
import glob
import subprocess
import matplotlib.pyplot as plt


def dataset_split(dataset_x,dataset_y,valid_n):
    train_x = dataset_x[:-valid_n,:]
    train_y = dataset_y[:-valid_n,:]
    valid_x = dataset_x[-valid_n:,:]
    valid_y = dataset_y[-valid_n:,:]
    return train_x, train_y, valid_x, valid_y


def date_iso(time_format='%Y%m%dT%H%M%S'):
    return datetime.now().strftime(time_format)


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
        filename_=f"{filename}-{date_iso()}.mp4"
        subprocess.call(f'ffmpeg -y -loglevel warning -framerate {framerate} -pattern_type glob -i *.png -c:v libx264 -pix_fmt yuv420p -vf crop=trunc(iw/2)*2:trunc(ih/2)*2 {filename_}'.split(' '))
        if rm_images: subprocess.call(['rm']+glob.glob("*.png"))
        print(f"created {filename_}")
    finally:
        os.chdir(previous_directory)


def matplotlib_settings(small_size=10,medium_size=14,bigger_size=16):
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title