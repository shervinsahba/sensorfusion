import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import optim, tensor
import torch.nn.functional as F

from .plots import matplotlib_settings, generate_video, mkdir_figs_vids, plot_2d_result, plot_psd
from .SDNN import SDNN, get_dataloader, predict
from .tools import dataset_split, demean, timethis
from .torch_boilerplate import fit, get_torchdevice

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, help='data.npy')
    parser.add_argument('--valid_n', type=int, nargs='?', default=500, help='n validation snapshots')
    parser.add_argument('--epochs', type=int, nargs='?', default=2000, help='training steps')
    parser.add_argument('--bs', type=int, nargs='?', default=512, help='batch size')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-4, help='learning rate')
    parser.add_argument('--l1', type=int, nargs='?', default=8192, help='layer 1 neurons')
    parser.add_argument('--l2', type=int, nargs='?', default=256, help='layer 2 neurons')
    parser.add_argument('--dp', type=float, nargs='?', default=0.3, help='dropout percentage')
    parser.add_argument('--devrun', action='store_true', help='dev mode quickrun')
    parser.add_argument('--makefigs', action='store_true', help='toggle making figures on')
    parser.add_argument('--makevids', action='store_true', help='toggle making movies on')
    return parser.parse_args()


def load_data(datafile: str) -> tuple:
    """Load data from file."""
    data_path = Path(f"data/ael/{datafile}")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")
    with open(data_path, 'rb') as f:
        dataset_x = np.load(f)
        dataset_y = np.load(f)
        shape_x = np.load(f)
        shape_y = np.load(f)
        en = np.load(f)
    return dataset_x, dataset_y, shape_x, shape_y, en


def prepare_data(dataset_x: np.ndarray, dataset_y: np.ndarray, valid_n: int) -> tuple:
    """Split and normalize datasets."""
    train_x, train_y, valid_x, valid_y = dataset_split(dataset_x, dataset_y, valid_n)
    train_x, train_y, valid_x, valid_y = map(demean, [train_x, train_y, valid_x, valid_y])
    print("train_x","train_y", [x.shape for x in [train_x, train_y]])
    print("valid_x","valid_y", [x.shape for x in [valid_x, valid_y]])
    return train_x, train_y, valid_x, valid_y


def train_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
                bs: int, lr: float, l1: int, l2: int, dp: float, epochs: int, devrun: bool) -> tuple:
    """Train the model."""
    dev = get_torchdevice()
    train_dl = get_dataloader(train_x, train_y, bs, shuffle=False)
    valid_dl = get_dataloader(valid_x, valid_y, bs * 2, shuffle=True)
    model = SDNN(train_x.shape[-1], train_y.shape[-1], l1=l1, l2=l2, dp=dp).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.mse_loss
    if devrun:
        epochs = 3
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    return model


def evaluate_model(model, train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray) -> tuple:
    """Evaluate the model and get predictions."""
    model.cpu()  # retrieve model to cpu
    train_r, train_loss = predict(train_x, train_y, model)
    valid_r, valid_loss = predict(valid_x, valid_y, model)
    return train_r, train_loss, valid_r, valid_loss


def reshape_data(train_x: np.ndarray, train_y: np.ndarray, train_r: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray, valid_r: np.ndarray,
                 en: int, shape_x: tuple, shape_y: tuple) -> tuple:
    """Reshape data and unstack training set."""
    train_x = train_x[:, :train_x.shape[1] // en]
    valid_x = valid_x[:, :valid_x.shape[1] // en]
    data_train = [
        train_x.reshape(-1, *shape_x[1:]),
        train_y.reshape(-1, *shape_y[1:]),
        train_r.reshape(-1, *shape_y[1:])
    ]
    data_valid = [
        valid_x.reshape(-1, *shape_x[1:]),
        valid_y.reshape(-1, *shape_y[1:]),
        valid_r.reshape(-1, *shape_y[1:])
    ]

    return data_train, data_valid


def generate_plots_and_videos(data_train: list, data_valid: list, figbasename: str, l1: int, l2: int,
                              makevids: bool):
    """Generate plots and videos."""
    mkdir_figs_vids()

    _, _, (vmin, vmax) = plot_2d_result(*data_train)
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

    if makevids:
        f = lambda t: plot_2d_result(*data_train, t=t, vmin=vmin, vmax=vmax)
        g = lambda t: plot_2d_result(*data_valid, t=t, vmin=vmin, vmax=vmax)
        generate_video(f, 50, start=0, directory='figs/vids/', framerate=6, filename=f"{figbasename}_train_{l1}_{l2}")
        generate_video(g, 50, start=0, directory='figs/vids/', framerate=6, filename=f"{figbasename}_valid_{l1}_{l2}")


@timethis
def main(datafile: str, valid_n: int, epochs: int, bs: int, lr: float, l1: int, l2: int, dp: float,
         devrun: bool = False, makefigs: bool = False, makevids: bool = False) -> tuple:
    """Run the main script."""
    dataset_x, dataset_y, shape_x, shape_y, en = load_data(datafile)
    train_x, train_y, valid_x, valid_y = prepare_data(dataset_x, dataset_y, valid_n)
    model = train_model(train_x, train_y, valid_x, valid_y, bs, lr, l1, l2, dp, epochs, devrun)
    train_r, train_loss, valid_r, valid_loss = evaluate_model(model, train_x, train_y, valid_x, valid_y)
    print(f"train_loss: {train_loss}")
    print(f"valid_loss: {valid_loss}")

    data_train, data_valid = reshape_data(train_x, train_y, train_r, valid_x, valid_y, valid_r,
                                          en, shape_x, shape_y)
    if makefigs or makevids:
        figbasename = Path(datafile).stem
        generate_plots_and_videos(data_train, data_valid, figbasename, l1, l2, makevids)

    return data_train, data_valid


if __name__ == "__main__":
    matplotlib_settings()
    args = parse_arguments()
    main(**vars(args))
