#!/usr/bin/env python3
import importlib.util
import json
# import tikzplotlib
import os
import sys

from seaborn.utils import despine

os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
import copy
import pandas as pd
import time

# import graphik
import matplotlib.pyplot as plt
import numpy as np
from generative_graphik.args.utils import str2bool
import argparse
import seaborn as sns
sns.set_theme(style="darkgrid")

def main(args):
    data = pd.read_pickle(f"{sys.path[0]}/results/{args.id}/results.pkl")
    stats = data.reset_index()
    stats["Err. Position"] = stats["Err. Position"]*1000
    stats["Err. Rotation"] = stats["Err. Rotation"]*(180/np.pi)
    q_pos = stats["Err. Position"].quantile(0.99)
    q_rot = stats["Err. Rotation"].quantile(0.99)
    stats = stats.drop(stats[stats["Err. Position"] > q_pos].index)
    stats = stats.drop(stats[stats["Err. Rotation"] > q_rot].index)

    stats = stats.groupby(["Robot", "Id"])[["Err. Position", "Err. Rotation"]].describe().groupby("Robot").mean()

    stats = stats.drop(["count", "std", "50%"], axis=1, level=1)

    perc_data = data.set_index(["Robot", "Id"])
    perc_data["Success"] = (
        (perc_data["Err. Position"] < 0.01) & (perc_data["Err. Rotation"] < (180/np.pi))
    )
    print(perc_data["Err. Position"])
    suc_pos_perc = (
        perc_data["Success"]
        .eq(True)
        .groupby(level=[0, 1])
        .value_counts(True)
        .unstack(fill_value=0)
    )
    stats["Success [\%]"] = suc_pos_perc.groupby(level=0).apply(lambda c: (c>0).sum()/len(c))[True]*100

    stats.rename(columns = {'75%': 'Q$_{3}$', '25%': 'Q$_{1}$','Err. Position':'Err. Pos. [mm]', 'Err. Rotation':'Err. Rot. [deg]'}, inplace = True)

    # Swap to follow paper order
    cols = stats.columns.tolist()
    ins = cols.pop(4)
    cols.insert(2, ins)
    ins = cols.pop(9)
    cols.insert(7, ins)
    stats = stats[cols]

    if args.save_latex:
        s = stats.style
        s.format(precision=1)
        s.format_index(axis=1,level=[0,1])
        latex = s.to_latex(hrules=True, multicol_align="c")
        print(latex)

        # open text file
        # text_file = open(args.latex_path + "tables/experiment_2.tex", "w")
        # write string to file
        # text_file.write(latex)

if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--save_latex", type=str2bool, default=True, help="Save latex table.")
    parser.add_argument("--latex_path", type=str, default="/home/filipmrc/Documents/Latex/2022-limoyo-maric-generative-corl/tables/experiment_2.tex", help="Base path for folder with experiment data")

    args = parser.parse_args()
    main(args)