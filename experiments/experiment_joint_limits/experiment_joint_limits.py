import importlib.util
import json
import os
import sys
import argparse
from graphik.graphs import ProblemGraphRevolute
from graphik.graphs.graph_revolute import list_to_variable_dict
from graphik.robots import RobotRevolute

from torch_geometric.data import DataLoader
from generative_graphik.utils.dataset_generation import generate_data_point

from generative_graphik.utils.torch_utils import batchFKmultiDOF, batchIKmultiDOF, node_attributes

os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
from argparse import Namespace
import copy
import pandas as pd
import time

import graphik
import matplotlib.pyplot as plt
import numpy as np
import torch
from generative_graphik.args.parser import parse_analysis_args
from graphik.utils.roboturdf import (
    RobotURDF,
    load_ur10,
    load_kuka,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_panda,
)
# import pyrender

from graphik.utils.dgp import graph_from_pos
from liegroups.numpy import SE3, SO3

def model_arg_loader(path):
    """Load hyperparameters from trained model."""
    if os.path.isdir(path):
        with open(os.path.join(path, "hyperparameters.txt"), "r") as fp:
            return Namespace(**json.load(fp))

# NOTE generates all the initializations and stores them to a pickle file
def main(args):
    device = args.device
    num_evals = args.n_evals  # number of evaluations
    robot_types = args.robots

    evals_per_robot = num_evals // len(robot_types)  
    for model_path in args.model_path:# number of evaluations per robots
        spec = importlib.util.spec_from_file_location("model", model_path + "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)

        # load models
        model_args = model_arg_loader(model_path)
        model = model.Model(model_args).to(device)
        name = model_args.id.replace("model", "results")
        c = np.pi / 180

        if model_path is not None:
            try:
                state_dict = torch.load(model_path + f"checkpoints/checkpoint.pth", map_location=device)
                model.load_state_dict(state_dict["net"])
                model.eval()
            except Exception as e:
                print(e)

        all_sol_data = []
        fig_handle, ax_handle = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        for robot_type in robot_types:
            if robot_type == "panda":
                limits_l = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
                limits_u = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
                limits = [limits_l, limits_u]
                robot, graph = load_panda(limits=None)
                fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
                urdf_robot = RobotURDF(fname)
            else:
                raise NotImplementedError

            for kdx in range(evals_per_robot):
                sol_data = []

                # Generate random problem
                prob_data = generate_data_point(graph).to(device)
                prob_data.num_graphs = 1
                # T_goal = prob_data.T_ee.cpu().numpy()
                data = model.preprocess(prob_data)
                P_goal = data.pos.cpu().numpy()
                T_goal = SE3.exp(data.T_ee.cpu().numpy())
                # T_goal = SE3.exp(data.T_ee[0].cpu().numpy())
                # q_goal = graph.joint_variables(graph_from_pos(P_goal, graph.node_ids))
                # q_goal_np = np.fromiter(
                #     (q_goal[f"p{jj}"] for jj in range(1, graph.robot.n + 1)), dtype=float
                # )

                P_all = model.forward_eval(data, num_samples=args.num_samples).cpu().detach().numpy()
                # Compute solutions
                # P_all = (
                #         model.forward_eval(
                #             x=data.pos, 
                #             h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1), 
                #             edge_attr=data.edge_attr, 
                #             edge_attr_partial=data.edge_attr_partial, 
                #             edge_index=data.edge_index_full, 
                #             partial_goal_mask=data.partial_goal_mask, 
                #             nodes_per_single_graph= int(data.num_nodes / 1),
                #             batch_size=1,
                #             num_samples=args.num_samples
                #         )
                # )

                # Analyze solutions
                e_pose = np.empty([P_all.shape[0]])
                e_pos = np.empty([P_all.shape[0]])
                e_rot = np.empty([P_all.shape[0]])
                q_sols_np = np.empty([P_all.shape[0], robot.n])
                for idx in range(P_all.shape[0]):
                    P = P_all[idx, :]
                    # q_sol = batchIKmultiDOF(
                    #     P, 
                    #     prob_data.T0, 
                    #     prob_data.num_joints, 
                    #     T_final = torch.tensor(T_goal.as_matrix(), dtype=P.dtype).unsqueeze(0).to(device)
                    # )

                    q_sol = graph.joint_variables(
                        graph_from_pos(P, graph.node_ids), {robot.end_effectors[0]: T_goal}
                    )  # get joint angles

                    q_sols_np[idx] = np.fromiter(
                        (q_sol[f"p{jj}"] for jj in range(1, graph.robot.n + 1)), dtype=float
                    )

                    T_ee = graph.robot.pose(q_sol, robot.end_effectors[-1])
                    e_pose[idx] = np.linalg.norm(T_ee.inv().dot(T_goal).log())
                    e_pos[idx] = np.linalg.norm(T_ee.trans - T_goal.trans)
                    e_rot[idx] = np.linalg.norm(T_ee.rot.inv().dot(T_goal.rot).log())

                    entry = {
                        "Id": kdx,
                        "Robot": robot_type,
                        "Goal Pose": T_goal.as_matrix(),
                        "Sol. Config": q_sols_np[idx],
                        # "Sol. Points": P_all[idx,:],
                        "Err. Pose": e_pose[idx],
                        "Err. Position": e_pos[idx],
                        "Err. Rotation": e_rot[idx],
                        # "Goal Config": q_goal_np,
                        # "Goal Points": P_goal,
                    }
                    sol_data.append(entry)
                all_sol_data.append(pd.DataFrame(sol_data))

        pd_data = pd.concat(all_sol_data)

        exp_dir = f"{sys.path[0]}/results/"+ f"{args.id}/"
        os.makedirs(exp_dir, exist_ok=True)
        pd_data.to_pickle(os.path.join(exp_dir, "results.pkl"))


if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--model_path", nargs="*", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument("--robots", nargs="*", type=str, default=["planar_chain"], help="Type of robot used")
    parser.add_argument("--n_evals", type=int, default=1000, help="Number of evaluations")
    parser.add_argument("--num_samples", type=int, default=1, help="Total number of samples per problem")

    args = parser.parse_args()
    main(args)