import argparse
import numpy as np
from collections import Counter
import h5py

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data types [shsh,dhdh,shdh]')
    parser.add_argument('--tr', help='temporal sampling rate', nargs='?', default=8, type=int)
    parser.add_argument('--xr', help='spatial sampling rate along 1st axis', nargs='?', default=3, type=int)
    parser.add_argument('--yr', help='spatial sampling rate along 2nd axis', nargs='?', default=3, type=int)
    parser.add_argument('--exp', help='ael experiment', nargs='?', default=0, type=int)                         
    parser.add_argument('--en', help='number of embeddings', nargs='?', default=1, type=int)                       
    return parser.parse_args()


def data_load(n, directory):
    """
    loads numpy matrices for sh and dh experiment n
    """
    h5f = h5py.File(f'{directory}experiment{n}.h5','r')
    
    # load wavefront OPD data and trim
    sh_phi = h5f['sh_phi'][:][4:-2,2:,:]
    dh_phi = h5f['dh_phi'][:][3:158,2:-1,:]
    
    # flip frame so it plots with the edge on the bottom
    sh_phi, dh_phi = map(np.flipud, [sh_phi,dh_phi])
    # move time axis to first position
    sh_phi, dh_phi = map(lambda x: np.moveaxis(x,-1,0), [sh_phi,dh_phi])
    
    # load time data
    sh_t = h5f['sh_t'][:]    
    dh_t = h5f['dh_t'][:]

    print(f"Loaded sh_phi {sh_phi.shape}, dh_phi {dh_phi.shape}, sh_t {sh_t.shape}, dh_t {dh_t.shape}")
    
    return sh_phi, dh_phi, sh_t, dh_t


def duplicates(x):
    """
    returns a list of duplicate entries in a list
    """
    return [k for (k,v) in Counter(list(x)).items() if v > 1]


def get_matching_indices(a, b, roundto=5):
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
    print(f"The number of matched times is {len(matches)}.")

    # get the indices of the matched times for each sensor's time vector
    a_idx = [j for j,t in enumerate(a_round) if t in matches]
    b_idx = [j for j,t in enumerate(b_round) if t in matches]

    return a_idx, b_idx


def stack_indices(idx, sr):
    idx = np.array(idx)
    c = np.empty(idx.size * sr, dtype=np.int32)
    for s in range(sr):
        c[s::sr] = idx + s
    return c


def stack_samples(a, b=None, embed=1):
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
        b = stack_samples(a,b,embed)
    return b


def main(data,tr,xr,yr,exp,en):

    # Select from experiments [0,1,2,3,4]
    sh_phi, dh_phi, sh_t, dh_t = data_load(exp, "data/ael/raw/")

    # trim spatial dimensions some more
    r = 3.46              # scale ratio
    # TODO the values really only apply to experiment 0.
    a,b,c,d = 4,18,17,45  # region of interest on SH sensor
    aa,bb,cc,dd = round(a*r),round(b*r),round((c-2)*r),round((d-2)*r)
    aa,bb,cc,dd = aa,bb+2,cc,dd+3 ### for dh to dh (50,150) shape
    sh_g = sh_phi[:,a:b,c:d]
    dh_g = dh_phi[:,aa:bb,cc:dd]
    print("trimmed sh and dh",[x.shape for x in [sh_g, dh_g]])

    # select data
    print(f"selecting data {data}")
    if data == "sh_sh":
        data_x = sh_g
        data_y = data_x
    elif data == "dh_dh":
        data_x = dh_g
        data_y = data_x
    elif data == "sh_dh":
        # there is no temporal sampling for sh_dh, so setting tr to match en
        # will work for any embeddings later
        tr = en  
        # trim dh times that are out of temporal range of the sh data
        if sh_t[-1] >= dh_t[-1]:
            dh_t = dh_t[dh_t<=sh_t[-1]]
            dh_g = dh_g[:len(dh_t),:,:]
        else:
            sh_t = sh_t[sh_t<=dh_t[-1]]
            sh_g = sh_g[:len(sh_t),:,:]
        # find matching time indices
        sh_idx, dh_idx = get_matching_indices(sh_t,dh_t,roundto=5) 
        # create embeddings with matched times
        dataset_x = sh_g[stack_indices(sh_idx,en),:,:]  # TODO bug that doesn't seem to affect production, but stack_indices can provide an out of range index theoretically
        dataset_y = dh_g[dh_idx,:,:]
        print("matched times for sh_g and dh_g",[x.shape for x in [dataset_x, dataset_y]])

    if data in ["sh_sh", "dh_dh"]:
        # subsample superset into x and y datasets
        dataset_x = data_x[:,::xr,::yr]
        dataset_y = data_y[::tr,:,:]
        print("subsampled", [x.shape for x in [dataset_x, dataset_y]], 
            f"with tr={tr}, xr={xr}, yr={yr}")

    # store shapes of original data for reconstruction
    shape_x = dataset_x.shape
    shape_y = dataset_y.shape

    # flatten frames to vecs
    dataset_x = dataset_x.reshape(dataset_x.shape[0],-1)
    dataset_y = dataset_y.reshape(dataset_y.shape[0],-1)
    print("vectorized", [x.shape for x in [dataset_x, dataset_y]])

    # create embeddings by stacking samples and then slicing
    dataset_x = stack_samples(dataset_x,embed=en)[::tr,:]
    # in case arrays differ in size, choose smaller and slice off excess
    n_snapshots = min(dataset_x.shape[0],dataset_y.shape[0])
    dataset_x = dataset_x[:n_snapshots,:]
    dataset_y = dataset_y[:n_snapshots,:]
    print("embeddings", [x.shape for x in [dataset_x, dataset_y]])

    # save dataset
    if data in ["sh_sh", "dh_dh"]:
        filename = f'data/ael/{data}_exp{exp}_tr{tr}_xr{xr}_yr{yr}_en{en}.npy'
    elif data == "sh_dh":
        filename = f'data/ael/{data}_exp{exp}_en{en}.npy'
    with open(filename, 'wb') as f:
        np.save(f, dataset_x)
        np.save(f, dataset_y)
        np.save(f, shape_x)
        np.save(f, shape_y)
    print(f"saved {filename}")


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))

