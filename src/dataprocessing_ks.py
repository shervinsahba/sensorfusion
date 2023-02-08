import scipy.io as sio
import numpy as np
import argparse

def stack_samples(a, b=None, embed=1):
    """creates row-wise embeddings of 2-D data.
    e.g. a (5,10) array with 3 embeddings returns a (15,8) array."""
    if b is None:
        b = a 
    if embed > 1:
        embed -= 1
        a = a[1:,:]
        b = np.hstack((b[:-1,:],a))                
        b = stack_samples(a,b,embed)
    return b


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr', help='temporal sampling rate', 
                        nargs='?', default=8, type=int)
    parser.add_argument('--xr', help='spatial sampling rate', 
                        nargs='?', default=64, type=int)
    parser.add_argument('--exp', help='ael experiment', 
                        nargs='?', default=0, type=int)                         
    parser.add_argument('--en', help='number of embeddings', 
                        nargs='?', default=1, type=int)          
    parser.add_argument('--label', help='label by xn or xr',
                        nargs='?', default='xr')              
    args = parser.parse_args()

    data = sio.loadmat('data/ks/raw/ks-superset-2048x98000.mat')['data']
    uu = data['uu'][0,0].T
    # tt = data['tt'][0,0].T

    # create embeddings by stacking samples and then slicing
    dataset_x = stack_samples(uu[:,::args.xr], embed=args.en)[::args.tr,:]
    dataset_y = uu[::args.tr,:]
    # in case arrays differ in size, choose smaller and slice off excess
    n_snapshots = min(dataset_x.shape[0],dataset_y.shape[0])
    dataset_x = dataset_x[:n_snapshots,:]
    dataset_y = dataset_y[:n_snapshots,:]
    print("embeddings", [x.shape for x in [dataset_x, dataset_y]])

    # save dataset
    xn = uu.shape[1]// args.xr   # low spatial resolution
    if args.label == 'xn':
        filename = f'data/ks/ks_tr{args.tr}_xn{xn}_en{args.en}.npy'
    elif args.label == 'xr':
        filename = f'data/ks/ks_tr{args.tr}_xr{args.xr}_en{args.en}.npy'
    with open(filename, 'wb') as f:
        np.save(f, dataset_x)
        np.save(f, dataset_y)
        np.save(f, xn)
    print(f"saved {filename}")    
