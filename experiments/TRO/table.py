#!/usr/bin/env python3
import json
import os
import sys
import importlib.util


os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
import copy
import pandas as pd
import time

# import graphik
import matplotlib.pyplot as plt
import numpy as np
from generative_graphik.args.parser import parse_analysis_args
from generative_graphik.args.utils import str2bool
import argparse
# import tikzplotlib
# import seaborn as sns
# from seaborn.utils import despine
# sns.set_theme(style="darkgrid")


def parse_analysis_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument(
        "--id",
        type=str,
        default="test_experiment",
        help="Name of the folder with experiment data",
    )
    parser.add_argument(
        "--save_latex",
        type=str2bool,
        default=True,
        help="Save latex table.",
    )
    parser.add_argument(
        "--latex_path",
        type=str,
        default="/home/filipmrc/Documents/Latex/2022-limoyo-maric-generative-corl/tables/experiment_2.tex",
        help="Base path for folder with experiment data",
    )

    args = parser.parse_args()
    return args


def main(args):
    # data = pd.read_pickle(f"{sys.path[0]}/results/{args.id}/results.pkl")
    data = pd.read_pickle(f"{sys.path[0]}/results/TRO/{args.id}/results.pkl")

    # # Smallest error samples per problem
    # min_pos_error_norm = data.groupby(["Robot", "Id"])["Err. Position"].min()
    # min_rot_error_norm = data.groupby(["Robot", "Id"])["Err. Rotation"].min()
    # min_pose_error_norm = data.groupby(["Robot", "Id"])["Err. Pose"].min()

    # # Percentage of samples per robot that have error lower than some criteria
    # suc_pos_perc_all = []
    # suc_rot_perc_all = []
    # perc_data = data.set_index(["Robot", "Id"])
    # pos_increment = 0.005
    # rot_increment = np.pi / 180
    # resolution = 1
    # for idx in [1, 5, 10]:
    #     perc_data["Suc. Pos"] = (
    #         perc_data["Err. Position"] < (idx / resolution) * pos_increment
    #     )
    #     suc_pos_perc = (
    #         perc_data["Suc. Pos"]
    #         .eq(True)
    #         .groupby(level=[0, 1])
    #         .value_counts(True)
    #         .unstack(fill_value=0)
    #     ) * 100
    #     suc_pos_perc["Error"] = (idx / resolution) * pos_increment * 100  # centimeters
    #     suc_pos_perc["Err. Type"] = "Pos"
    #     suc_pos_perc_all.append(suc_pos_perc)

    #     perc_data["Suc. Rot"] = (
    #         perc_data["Err. Rotation"] < (idx / resolution) * rot_increment
    #     )
    #     suc_rot_perc = (
    #         perc_data["Suc. Rot"]
    #         .eq(True)
    #         .groupby(level=[0, 1])
    #         .value_counts(True)
    #         .unstack(fill_value=0)
    #     ) * 100
    #     suc_rot_perc["Error"] = idx / resolution  # degrees
    #     suc_rot_perc["Err. Type"] = "Rot"
    #     suc_rot_perc_all.append(suc_rot_perc)


    # # Average lowest errors per robot
    # avg_min_pos_err_norm = min_pos_error_norm.groupby("Robot").mean()*100
    # avg_min_pose_err_norm = min_pose_error_norm.groupby("Robot").mean()
    # avg_min_rot_err_norm = min_rot_error_norm.groupby("Robot").mean()*100

    # # Average time it took to generate the samples
    # avg_sol_t = data.groupby("Robot")["Sol. Time"].mean()

    # # Table data
    # data_dict = {
    #     "Avg. Pos. Err.": avg_min_pos_err_norm,
    #     # "$\bar{e_{pos}}$": avg_min_pos_err_norm,
    #     # "Avg. Pose Err.": avg_min_pose_err_norm,
    #     "Avg. Rot. Err.": avg_min_rot_err_norm,
    #     "Avg. Sol. Time": avg_sol_t,
    # }

    # suc_perc_concat = pd.concat(suc_pos_perc_all + suc_rot_perc_all).reset_index()
    # for err_cut in sorted(suc_perc_concat["Error"].unique()):
    #     if err_cut in suc_perc_concat[suc_perc_concat["Err. Type"] == "Pos"]["Error"].unique():
    #         data_dict[f"Err. Pos. $<$ {err_cut}"] = suc_perc_concat[
    #             (suc_perc_concat["Err. Type"] == "Pos") & (suc_perc_concat["Error"] == err_cut)
    #         ].groupby("Robot")[True].mean()

    #     if err_cut in suc_perc_concat[suc_perc_concat["Err. Type"] == "Rot"]["Error"].unique():
    #         data_dict[f"Err. Rot. $<$ {err_cut}"] = suc_perc_concat[
    #             (suc_perc_concat["Err. Type"] == "Rot") & (suc_perc_concat["Error"] == err_cut)
    #         ].groupby("Robot")[True].mean()

    # # Assemble into new DataFrame
    # pd_all = pd.DataFrame(data_dict)
    # print(pd_all)
    # pd_all.index.name = None

    # # Format data
    # format_dict = {
    #             "Avg. Pos. Err.": "{:.2f}",
    #             "Avg. Pose Err.": "{:.2f}",
    #             "Avg. Rot. Err.": "{:.2f}",
    #             "Avg. Sol. Time": "{:.2f}",
    #         }
    # for err_cut in suc_perc_concat["Error"].unique():
    #     format_dict[f"Err. Pos. $<$ {err_cut}"] = "{:.1f}"
    #     format_dict[f"Err. Rot. $<$ {err_cut}"] = "{:.1f}"

    # # Print to latex table
    # if args.save_latex:
    #     s = pd_all.style
    #     s.format(format_dict)
    #     latex = s.to_latex(hrules=True)
    #     print(latex)

    #     # open text file
    #     text_file = open(args.latex_path + "tables/experiment_2.tex", "w")

    #     # write string to file
    #     text_file.write(latex)

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
    args = parse_analysis_args()
    main(args)
