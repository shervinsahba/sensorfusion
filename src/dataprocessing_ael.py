import argparse
import numpy as np
from collections import Counter
import h5py
from .tools import embed_snapshots, vectorize_frames


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data, one of [sh_sh,dh_dh,sh_dh]')
    parser.add_argument('--tr', help='temporal sampling rate', nargs='?', default=10, type=int)
    parser.add_argument('--xr', help='spatial sampling rate along 1st axis', nargs='?', default=5, type=int)
    parser.add_argument('--yr', help='spatial sampling rate along 2nd axis', nargs='?', default=5, type=int)
    parser.add_argument('--exp', help='ael experiment', nargs='?', default=0, type=int)                         
    parser.add_argument('--en', help='number of embeddings', nargs='?', default=1, type=int)                       
    return parser.parse_args()


def data_load(directory, exp=0):
    """
    loads numpy matrices for sh and dh experiment n
    """
    h5f = h5py.File(f'{directory}experiment{exp}.h5','r')
    
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


def data_select(sh_phi, dh_phi, sh_t, dh_t, exp=0):
    # Trim spatial dimensions some more.
    r = 3.46              # scale ratio
    # TODO these values might only apply to experiment 0.
    a,b,c,d = 4,19,15,45  # (15,30) region of interest on SH sensor
    aa,bb,cc,dd = map(round, r * np.array([a,b,c-2,d-2]))
    sh_g = sh_phi[:,a:b,c:d]
    dh_g = dh_phi[:,aa:bb-2,cc:dd-4]  # edit DH a bit for (50,100) shape

    # make sure our arrays don't contain nans
    if np.isnan(sh_g).any() or np.isnan(dh_g).any():
        raise ValueError("sh_g or dh_g contains nan! choose a different region.")

    # Trim dh/sh times that are out of temporal range of the sh/dh data.
    if sh_t[-1] >= dh_t[-1]:
        dh_t = dh_t[dh_t<=sh_t[-1]]
        dh_g = dh_g[:len(dh_t),:,:]
    else:
        sh_t = sh_t[sh_t<=dh_t[-1]]
        sh_g = sh_g[:len(sh_t),:,:]
    
    return sh_g, dh_g, sh_t, dh_t


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
    print(f"found {len(matches)} matching times")

    # get the indices of the matched times for each sensor's time vector
    a_idx = [j for j,t in enumerate(a_round) if t in matches]
    b_idx = [j for j,t in enumerate(b_round) if t in matches]

    return a_idx, b_idx


def stack_indices(idx, sr):
    """ Takes indices and inserts subsequent integers, sr times.
    e.g. stack_indices([1,3,3,4],2) = array([1, 2, 3, 4, 3, 4, 4, 5], dtype=int32)
    e.g. stack_indices([1,3,7],3) = array([1, 2, 3, 3, 4, 5, 7, 8, 9], dtype=int32)
    """
    idx = np.array(idx)
    c = np.empty(idx.size * sr, dtype=np.int32)
    for s in range(sr):
        c[s::sr] = idx + s
    return c


def main(data,tr,xr,yr,exp,en):
    if data not in ["sh_sh","dh_dh","sh_dh"]:
        raise ValueError("data needs to be one of [sh_sh,dh_dh,sh_dh]")
    
    # Select from experiments [0,1,2,3,4]
    sh_phi, dh_phi, sh_t, dh_t = data_load("data/ael/raw/", exp=exp)
    sh_g, dh_g, sh_t, dh_t = data_select(sh_phi, dh_phi, sh_t, dh_t, exp=exp)

    print("selected",[x.shape for x in [sh_g, dh_g]])

    # select data
    dataset_x = sh_g if data in ["sh_sh","sh_dh"] else dh_g
    dataset_y = sh_g if data in ["sh_sh"] else dh_g
    if data == "sh_dh":
        # find matching time indices
        sh_idx, dh_idx = get_matching_indices(sh_t,dh_t,roundto=5) 

        # create datasets with matched times and desired temporal stacking
        dataset_x = dataset_x[stack_indices(sh_idx,en),:,:]  # TODO bug that doesn't seem to affect production: stack_indices can provide an out of range index if the end of the temporal range is used with a lot of embeddings.
        dataset_y = dataset_y[dh_idx,:,:]
        print(f"stacked {en}x", [x.shape for x in [dataset_x, dataset_y]])
        
        # The temporal embedding isn't complete here. We inserted the 
        # needed snapshots, which multiplies the number of snapshots by en.
        # Later we will need to append these additional snapshots to the originals,
        # completing the embedding. This will be done by embed_snapshots,
        # which will not only embed snapshots for each matched time, but will also
        # embed en-1 more times than necessary.
        #
        # Remember there is no temporal sampling being done to sh_dh data.
        # So we use the tr parameter to select every en snapshots, completing the
        # embedding and retaining only the proper snapshots.
        tr = en  
    else:
        # subsample superset into x and y datasets
        dataset_x = dataset_x[:,::xr,::yr]  # low spatial / high temporal resolution
        dataset_y = dataset_y[::tr,:,:]     # high spatial/ low temporal resolution
        print("subsampled", [x.shape for x in [dataset_x, dataset_y]], 
            f"with tr={tr}, xr={xr}, yr={yr}")

    # store shapes of original datasets for reconstruction, then flatten frames to vecs
    shape_x = dataset_x.shape
    shape_y = dataset_y.shape
    dataset_x, dataset_y = map(vectorize_frames,[dataset_x, dataset_y])
    print("vectorized", [x.shape for x in [dataset_x, dataset_y]])

    # create embeddings by stacking samples and then slicing
    dataset_x = embed_snapshots(dataset_x,embed=en)[::tr,:]
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

