import scipy.io as sio
import numpy as np
import argparse
from .tools import embed_snapshots

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ks simulation data', nargs='?', default='data/ks/raw/ks-superset-2048x98000.mat')
    parser.add_argument('--tr', help='temporal sampling rate', nargs='?', default=8, type=int)
    parser.add_argument('--xr', help='spatial sampling rate', nargs='?', default=64, type=int)
    parser.add_argument('--exp', help='ael experiment', nargs='?', default=0, type=int)                         
    parser.add_argument('--en', help='number of embeddings', nargs='?', default=1, type=int)          
    parser.add_argument('--label', help='label by xn or xr', nargs='?', default='xr')              
    return parser.parse_args()


def main(data,tr,xr,exp,en,label):

    simulation_data = sio.loadmat(data)['data']
    uu = simulation_data['uu'][0,0].T
    # tt = data['tt'][0,0].T

    # subsample superset into x and y datasets
    dataset_x = uu[:,::xr]
    dataset_y = uu[::tr,:]
    shape_x = dataset_x.shape
    shape_y = dataset_y.shape

    # create embeddings by stacking samples and then slicing
    dataset_x = embed_snapshots(dataset_x, embed=en)[::tr,:]
    
    # in case arrays differ in size, choose smaller and slice off excess
    n_snapshots = min(dataset_x.shape[0],dataset_y.shape[0])
    dataset_x = dataset_x[:n_snapshots,:]
    dataset_y = dataset_y[:n_snapshots,:]
    print("embeddings", [x.shape for x in [dataset_x, dataset_y]])

    # save dataset
    filename_formats = {
        'xn': f'data/ks/ks_tr{tr}_xn{dataset_x.shape[1]}_en{en}.npy',
        'xr': f'data/ks/ks_tr{tr}_xr{xr}_en{en}.npy'
    }
    filename = filename_formats[label]
    with open(filename, 'wb') as f:
        np.save(f, dataset_x)
        np.save(f, dataset_y)
        np.save(f, shape_x)
        np.save(f, shape_y)
        np.save(f, en)
    print(f"saved {filename}")    


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))