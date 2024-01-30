#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import tikzplotlib
import random
import argparse
import pickle as pkl
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set(font_scale=1.65, rc={'text.usetex' : True, "font.family": "Computer Modern"})
params = ["xmajorticks=true", "ymajorticks=true", "xtick pos=left", "ytick pos=left"]
import pandas as pd

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def main(args):
    path = f"{sys.path[0]}/results/{args.id}/results.pkl"
    
    with open(path, 'rb') as f:
        data = pkl.load(f)
    data = np.array(data, dtype=object)
    data[:, 1] = data[:, 1] * 1000
    df = pd.DataFrame(data, columns=["Number of Sampled Configurations", "Time [ms]", "DOF"])
    out = sns.lineplot(data=df, x="Number of Sampled Configurations", y="Time [ms]", hue="DOF", errorbar="sd")
    # plt.yscale('log')
    fig = out.get_figure()

    plt.tight_layout()
    tikzplotlib_fix_ncols(fig)
    
    save_path = os.path.join(args.save_path, f"{args.id}/plots/")
    os.makedirs(save_path, exist_ok=True)
    tikzplotlib.save(
        os.path.join(save_path, "timing_plot.tex"),
        figure="gcf",
        textsize=12.0,
        extra_axis_parameters=params,
        wrap=False,
    )
    
    fig.savefig(
        os.path.join(save_path, "timing_plot.pdf")
    )

if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for PyTorch')
    
    args = parser.parse_args()
    main(args)
