import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from .plots import matplotlib_settings, generate_video, mkdir_figs_vids, plot_1d_results
from .SDNN import train_model, evaluate_model
from .tools import load_data, prepare_data, reshape_1d_data

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, help='data.npy')
    parser.add_argument('--valid_n', type=int, nargs='?', default=500, help='n validation snapshots')
    parser.add_argument('--epochs', type=int, nargs='?', default=100, help='training steps')
    parser.add_argument('--bs', type=int, nargs='?', default=128, help='batch size')
    parser.add_argument('--lr', type=float, nargs='?', default=0.002, help='learning rate')
    parser.add_argument('--l1', type=int, nargs='?', default=16384, help='layer 1 neurons')
    parser.add_argument('--l2', type=int, nargs='?', default=64, help='layer 2 neurons')
    parser.add_argument('--dp', type=float, nargs='?', default=0.3, help='dropout percentage')
    parser.add_argument('--devrun', action='store_true', help='dev mode quickrun')
    parser.add_argument('--makefigs', action='store_true', help='toggle making figures on')
    parser.add_argument('--makevids', action='store_true', help='toggle making movies on')
    return parser.parse_args()


def generate_plots_and_videos(data_valid: list, figbasename: str, makevids: bool):
    """Generate plots and videos."""
    mkdir_figs_vids()

    print(f"Making figure {figbasename}.png")
    plot_1d_results(*data_valid)
    plt.savefig(f"figs/{figbasename}.png", transparent=False, bbox_inches='tight', dpi=300)

    if makevids:
        print("Making video...")
        plot_to_frame = lambda t: plot_1d_results(*data_valid, inset_snapshot_t=t)
        generate_video(plot_to_frame, 475, start=25, directory='figs/vids', filename=f"{figbasename}")


def main(datafile: str, valid_n: int, epochs: int, bs: int, lr: float, l1: int, l2: int, dp: float,
         devrun: bool = False, makefigs: bool = False, makevids: bool = False) -> tuple:
    """Run the main script."""
    dataset_x, dataset_y, shape_x, shape_y, en = load_data(datafile, "data/ks/")
    train_x, train_y, valid_x, valid_y = prepare_data(dataset_x, dataset_y, valid_n)
    model = train_model(train_x, train_y, valid_x, valid_y, bs, lr, l1, l2, dp, epochs, devrun)
    train_r, train_loss, valid_r, valid_loss = evaluate_model(model, train_x, train_y, valid_x, valid_y)
    print(f"train_loss: {train_loss}")
    print(f"valid_loss: {valid_loss}")

    data_train = reshape_1d_data(train_x, train_y, train_r, shape_x)
    data_valid = reshape_1d_data(valid_x, valid_y, valid_r, shape_x)

    if makefigs or makevids:
        figbasename =f"{Path(datafile).stem}_valid_{l1}_{l2}"
        generate_plots_and_videos(data_valid, figbasename, makevids)

    return data_train, data_valid


if __name__ == "__main__":
    matplotlib_settings()
    args = parse_arguments()
    main(**vars(args))
