from collections import Counter
from pathlib import Path
import numpy as np
import functools
import time

## decorators

def along_axis(func,func_ndim=2):
    """Creates a @along_axis(func_ndim) decorator that expands a numpy function
    that actions on an ndim array to also operate on ndim+1 arrays.
    """
    @functools.wraps(func)
    def wrapper(arr, *args, axis=0, **kwargs):
        if arr.ndim == func_ndim:
            return func(arr,*args,**kwargs)
        elif arr.ndim == func_ndim + 1:
            slices_pre = [slice(None, None) for i in range(axis)]
            slices_post = [slice(None, None) for i in range(arr.ndim-axis-1)]
            return np.array([func(arr[(*slices_pre, j, *slices_post)],*args,**kwargs) for j in range(arr.shape[axis])])
        else:
            raise TypeError(f"input array has ndim {arr.ndim}, but {func.__name__} requires dimensionality {func_ndim} or, with an axis specified, {func_ndim+1}.")
    return wrapper


def timethis(func):
    """Creates a @timethis decorator. As long as timethis is in the namespace,
    you can put the @timethis decorator preceding any function to print its walltime.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("time:", func.__name__, round(end-start, 6), "s")
        return result
    return wrapper


## functions

def dataset_split(dataset_x,dataset_y,valid_n):
    train_x = dataset_x[:-valid_n,:]
    train_y = dataset_y[:-valid_n,:]
    valid_x = dataset_x[-valid_n:,:]
    valid_y = dataset_y[-valid_n:,:]
    return train_x, train_y, valid_x, valid_y


def demean(a):
    return (a - a.mean(0))


def duplicates(x):
    """
    returns a list of duplicate entries in a list
    """
    return [k for (k,v) in Counter(list(x)).items() if v > 1]


def embed_snapshots(a, b=None, embed=1):
    """Creates row-wise embeddings of 2-D data.
    Each row is appended to the previous row. The final row is discarded.
    example: inputting a (5,10) array with embed=3 returns a (3,30) array.
    Creating temporal embeddings assumes the first dimension in the array is time."""
    if b is None:
        b = a
    if embed > 1:
        embed -= 1
        a = a[1:,:]
        b = np.hstack((b[:-1,:],a))
        b = embed_snapshots(a,b,embed)
    return b


def expand_hack(x, y, z=None, gap=1e-3):
    """This is a dirty hack used to plot "points" using plt.plot
    instead of plt.scatter. 
    
    This creates pairs of elements, x[i] and x[i]+gap, separated by np.nan.
    Because plt.plot ignores the np.nan entries, this can create the
    illusion of points. This is useful because these points, unlike
    markers from plt.scatter, are plotted as a single object
    and thus have a single, non-overlapping transparency.
    see: https://stackoverflow.com/a/50203803
    """
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    if z is not None:
        z1 = np.repeat(z, 3) + add
        return x1, y1, z1
    return x1, y1


def load_data(datafile: str, datadir: str) -> tuple:
    """Load data from file."""
    data_path = Path(f"{datadir}{datafile}")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")
    with open(data_path, 'rb') as f:
        dataset_x = np.load(f)
        dataset_y = np.load(f)
        shape_x = np.load(f)
        shape_y = np.load(f)
        en = np.load(f)
    return dataset_x, dataset_y, shape_x, shape_y, en


def matching_search_unique(a, b, roundto=5):
    """
    returns indices of unique matching entries between two numerical lists, a and b.
    """
    # round time vectors to specified decimal
    a_round = np.round(a, roundto)
    b_round = np.round(b, roundto)

    # make sure all times are unique within each vector
    if duplicates(a_round) or duplicates(b_round):
        print("warning: there are duplicate entries within each vector, \
        possibly from excess rounding.")
    
    # combine time vectors as lists. find which times occurred in both lists.
    matches = duplicates(list(a_round) + list(b_round))
    print(f"found {len(matches)} matching times")

    # get the indices of the matched times for each sensor's time vector
    a_idx = [j for j,t in enumerate(a_round) if t in matches]
    b_idx = [j for j,t in enumerate(b_round) if t in matches]

    return a_idx, b_idx


def matching_search_nearby(x,y):
    """For each element of y, return the indices of the closest values in x.

    Args:
        x (ndarray): 1-D vector to be searched
        y (ndarray): 1-D vector whose elements you wish to match to those in x

    Returns:
        ndarry: indices for the matched search

    Raises:
        ValueError: if np.max(y) > np.max(x) the search will return an out of bounds index.
    """
    idx_sorted = np.argsort(x)   # get sorted indices of x (in case not sorted)
    x = x[idx_sorted]            # sort x

    if np.max(y) > np.max(x):
        raise ValueError(f"Max value of matching array {np.max(y)} exceeds \
            max value of searched array {np.max(x)}. This is destined to fail. \
            Truncate the matching array or switch the roles of the inputs.")

    # find indices into sorted array x such that if corresponding elements
    # of y were inserted before these indices, order would be preserved
    idx1 = np.searchsorted(x,y)
    
    # get another set of indices if we placed the element to left of
    # where idx1 would have placed it (i.e. idx1 minus 1 if possible)
    idx2 = np.clip(idx1 - 1, 0, len(x)-1)

    # compute differences between idx1 and idx2 placement. keep lowest.
    idx_match = np.where(x[idx1] - y <= y - x[idx2], idx1, idx2)

    return idx_sorted[idx_match]


def mse_snapshots(a,b,axis=1):
    return np.mean((a-b)**2,axis=axis)


def prepare_data(dataset_x: np.ndarray, dataset_y: np.ndarray, valid_n: int) -> tuple:
    """Split and normalize datasets."""
    train_x, train_y, valid_x, valid_y = dataset_split(dataset_x, dataset_y, valid_n)
    train_x, train_y, valid_x, valid_y = map(demean, [train_x, train_y, valid_x, valid_y])
    print("train_x","train_y", [x.shape for x in [train_x, train_y]])
    print("valid_x","valid_y", [x.shape for x in [valid_x, valid_y]])
    return train_x, train_y, valid_x, valid_y


def psd2(data, fftshift=True, log=True):
    """Compute 2D power spectrum for an n x m array 
    or a t x n x m time series of (n x m) arrays.
    """
    axes = [1,2] if data.ndim ==3 else [0,1]
    result = np.abs(np.fft.fftn(data,axes=axes))**2
    if fftshift:
        result = np.fft.fftshift(result,axes=axes)
    if log:
        result = np.log(result)
    return result


def vectorize_frames(x):
    """ Takes an N-D array and returns a 2-D array with the first axis preserved.
    """
    return x.reshape(x.shape[0],-1)


def radial_mesh(image,origin='center'):
    if origin == 'center':
        origin = np.array(image.shape[::-1])//2
    X,Y = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
    return np.sqrt((X-origin[0])**2 + (Y-origin[1])**2)


@along_axis
def radial_mean(image, r=None, origin='center'):
    R = radial_mesh(image,origin=origin)
    f = np.vectorize(lambda r : image[(R >= r-0.5) & (R < r+0.5)].mean())
    # in most cases, we want results for all r in R
    if r is None:
        r  = np.arange(int(R.max()))
    return f(r)

@along_axis
def radial_psd(image, r=None, log=True, origin='center'):
    psd = psd2(image, log=log)
    return radial_mean(psd, r=r, origin=origin)


def reshape_1d_data(x: np.ndarray, y: np.ndarray, r: np.ndarray, 
                    shape_x: tuple) -> tuple:
    """Reshape data and unstack training set."""
    x = x[:, :shape_x[1]]
    return x, y, r


def reshape_2d_data(x: np.ndarray, y: np.ndarray, r: np.ndarray,
                 shape_x: tuple, shape_y: tuple) -> tuple:
    """Reshape data and unstack training set."""
    x = x[:, :np.product(shape_x[1:])]
    x = x.reshape(-1, *shape_x[1:])
    y = y.reshape(-1, *shape_y[1:])
    r = r.reshape(-1, *shape_y[1:])
    return x, y, r
