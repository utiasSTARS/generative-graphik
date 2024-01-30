import random
import argparse
import numpy as np
import os 
import sys 
import math
import pickle as pkl

import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set(font_scale=2.00, rc={'text.usetex' : True, "font.family": "Computer Modern"})
params = ["xmajorticks=true", "ymajorticks=true", "xtick pos=left", "ytick pos=left"]
import matplotlib.pyplot as plt

def main(args):
    path = f"{sys.path[0]}/results/{args.id}/results.pkl"
    with open(path, 'rb') as f:
        data = pkl.load(f)

    dataset = {
        'A': [],
        'B': [],
        'C': [],
        'D': [],
        'E': [],
        'F': [],
        'G': []
    }
        
    for q in data.get('Sol. Config'):
        for i, key in enumerate(dataset.keys()):
            dataset[key].append(q[i])

    save_path = os.path.join(args.save_path, f"{args.id}/plots/")
    os.makedirs(save_path, exist_ok=True)

    limits_l = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    limits_u = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    plt.figure(figsize=(10, 6))  # Adjust the width (10) as per your requirement
    df = pd.DataFrame(dataset)

    graph = sns.stripplot(x="variable", y="value", data=df.melt(), jitter=0.225, orient="v", s =3)
    # Get unique categories on x-axis and their positions
    categories = df.columns
    category_positions = np.arange(len(categories))
    category_width = 0.6  # Adjust this width to control the width of the shaded rectangle

    # Draw horizontal lines at y-values specified by limits_l and limits_u for each category
    for i, (limit_l, limit_u) in enumerate(zip(limits_l, limits_u)):
        tick_center = category_positions[i]  # Center of the current tick        
        # Shading the area between the dashed lines
        plt.fill_between(
            [tick_center - category_width/2, tick_center + category_width/2], 
            limit_l, 
            limit_u, 
            color='green', 
            alpha=0.25
        )

    plt.xticks(ticks=category_positions, labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$", r"$\theta_6$", r"$\theta_7$"])
    plt.xlabel("")  # Remove x-axis label
    plt.ylabel("Joint Angle [rads]")  # Rename y-axis label
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, "joint_limits.pdf")
    )

if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for PyTorch')
    
    args = parser.parse_args()
    main(args)
