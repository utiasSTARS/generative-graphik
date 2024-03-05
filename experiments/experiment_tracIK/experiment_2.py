import importlib.util
import json
import os
import sys
from graphik.graphs import ProblemGraphRevolute
from graphik.graphs.graph_revolute import list_to_variable_dict
from graphik.robots import RobotRevolute

from torch_geometric.data import DataLoader
from generative_graphik.utils.dataset_generation import generate_data_point

from generative_graphik.utils.torch_utils import SE3_inv, batchFKmultiDOF, batchIKmultiDOF, node_attributes, torch_log_from_T

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

def filter_by_distance(P_all, data, norm = 'inf'):
    src, dst = data["edge_index_full"]
    dist_samples = ((P_all[:,src] - P_all[:,dst])**2).sum(dim=-1).sqrt()
    dist_diff = dist_samples[:, data.partial_mask] - data["edge_attr_partial"].t()

    if norm == 'inf':
        dist_diff_norm, _ = torch.max(torch.abs(dist_diff), dim=-1)
    else:
        dist_diff_norm = (dist_diff**2).sum(dim=-1).sqrt()

    ind = torch.argsort(dist_diff_norm)
    return ind

def filter_by_error(T_ee, T_final_inv):
    e_pose = torch_log_from_T(torch.bmm(T_final_inv.expand(T_ee.shape[0],-1,-1), T_ee))
    e_pose_norm = torch.norm(e_pose, dim=-1)
    ind = torch.argsort(e_pose_norm)
    return ind


# NOTE generates all the initializations and stores them to a pickle file
def main(args):
    device = args.device
    num_evals = args.n_evals  # number of evaluations
    robot_types = args.robots
    dofs = args.dofs  # number of dof we test on
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
                # model.load_state_dict(
                #     torch.load(model_path + f"/net.pth", map_location=device)
                # )
                model.eval()
            except Exception as e:
                print(e)

        all_sol_data = []
        fig_handle, ax_handle = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        for robot_type in robot_types:
            if robot_type == "ur10":
                robot, graph = load_ur10(limits=None)
                fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
                urdf_robot = RobotURDF(fname)

                # # UR10 coordinates for testing
                # modified_dh = False
                # a = [0, -0.612, 0.5723, 0, 0, 0]
                # d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
                # al = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
                # # th = [0, np.pi, 0, 0, 0, 0]
                # th = [0, 0, 0, 0, 0, 0]

                # params = {
                #     "a": a,
                #     "alpha": al,
                #     "d": d,
                #     "theta": th,
                #     "modified_dh": modified_dh,
                #     "num_joints": 6,
                # }
                # robot = RobotRevolute(params)
                # graph = ProblemGraphRevolute(robot)
            elif robot_type == "kuka":
                limits_l = -np.array([170, 120, 170, 120, 170, 120, 170]) * c
                limits_u = np.array([170, 120, 170, 120, 170, 120, 170]) * c
                limits = [limits_l, limits_u]
                robot, graph = load_kuka(limits=None)
                fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
                urdf_robot = RobotURDF(fname)

                # modified_dh = False
                # a = [0, 0, 0, 0, 0, 0, 0]
                # d = [0.34, 0, 0.40, 0, 0.40, 0, 0.126]
                # al = [-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]
                # th = [0, 0, 0, 0, 0, 0, 0]

                # params = {
                #     "a": a,
                #     "alpha": al,
                #     "d": d,
                #     "theta": th,
                #     "modified_dh": modified_dh,
                #     "num_joints": 7,
                # }
                # robot = RobotRevolute(params)
                # graph = ProblemGraphRevolute(robot)
            elif robot_type == "lwa4d":
                limits_l = -np.array([180, 123, 180, 125, 180, 170, 170]) * c
                limits_u = np.array([180, 123, 180, 125, 180, 170, 170]) * c
                limits = [limits_l, limits_u]
                robot, graph = load_schunk_lwa4d(limits=None)
                fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
                urdf_robot = RobotURDF(fname)

                # modified_dh = False
                # a = [0, 0, 0, 0, 0, 0, 0]
                # d = [0.3, 0, 0.328, 0, 0.323, 0, 0.0824]
                # al = [-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, 0]
                # th = [0, 0, 0, 0, 0, 0, 0]

                # params = {
                #     "a": a,
                #     "alpha": al,
                #     "d": d,
                #     "theta": th,
                #     "modified_dh": modified_dh,
                #     "num_joints": 7,
                # }
                # robot = RobotRevolute(params)
                # graph = ProblemGraphRevolute(robot)
            elif robot_type == "panda":
                limits_l = -np.array(
                    [2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973]
                )
                limits_u = np.array(
                    [2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973]
                )
                limits = [limits_l, limits_u]
                robot, graph = load_panda(limits=None)
                fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
                urdf_robot = RobotURDF(fname)

                # modified_dh = False
                # a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
                # d = [0.333, 0, 0.316, 0, 0.384, 0, 0]
                # al = [0, -np.pi/2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
                # th = [0, 0, 0, 0, 0, 0, 0]

                # params = {
                #     "a": a,
                #     "alpha": al,
                #     "d": d,
                #     "theta": th,
                #     "modified_dh": modified_dh,
                #     "num_joints": 7,
                # }
                # robot = RobotRevolute(params)
                # graph = ProblemGraphRevolute(robot)
            elif robot_type == "lwa4p":
                limits_l = -np.array([170, 170, 155.3, 170, 170, 170]) * c
                limits_u = np.array([170, 170, 155.3, 170, 170, 170]) * c
                limits = [limits_l, limits_u]
                robot, graph = load_schunk_lwa4p(limits=None)
                fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
                urdf_robot = RobotURDF(fname)

                # modified_dh = False
                # a = [0, 0.350, 0, 0, 0, 0]
                # d = [0.205, 0, 0, 0.305, 0, 0.075]
                # al = [-np.pi / 2, np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]
                # th = [0, 0, 0, 0, 0, 0]

                # params = {
                #     "a": a,
                #     "alpha": al,
                #     "d": d,
                #     "theta": th,
                #     "modified_dh": modified_dh,
                #     "num_joints": 6,
                # }
                # robot = RobotRevolute(params)
                # graph = ProblemGraphRevolute(robot)
            else:
                raise NotImplementedError

            for kdx in range(evals_per_robot):
                sol_data = []

                # Generate random problem
                prob_data = generate_data_point(graph).to(device)
                prob_data.num_graphs = 1
                data = model.preprocess(prob_data)
                num_samples_pre = args.num_samples[0]*4

                T_goal = SE3.exp(data.T_ee.cpu().numpy())
                T_final = torch.tensor(T_goal.as_matrix(), dtype=torch.float32).unsqueeze(0).to(device)
                T_final_inv = torch.tensor(T_goal.inv().as_matrix(), dtype=torch.float32).unsqueeze(0).to(device)
                T_goal = T_goal.as_matrix()
                ee_ind = torch.cumsum(prob_data.num_joints.expand(num_samples_pre) + 1, dim=0) - 1 # end indices of joints

                data.goal_data_repeated_per_node = torch.repeat_interleave(1, 2*data.num_joints + model.num_anchor_nodes, dim=0)
                t0 = time.time()
                # Compute solutions
                # P_all = (
                #     model.forward_eval(data, num_samples=num_samples_pre).to(device).detach()
                # )
                P_all = model.forward_eval(
                    x=data.pos,
                    h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1),
                    edge_attr=data.edge_attr,
                    edge_attr_partial=data.edge_attr_partial,
                    edge_index=data.edge_index_full,
                    partial_goal_mask=data.partial_goal_mask,
                    nodes_per_single_graph = int(data.num_nodes / 1),
                    num_samples=num_samples_pre,
                    batch_size=data.num_graphs
                )

                # P_all = filter_by_distance(P_all, data, args.num_samples[0])
                # torch.cuda.synchronize()

                # q_sols_ = batchIKmultiDOF(
                #     P_all.reshape(-1,3),
                #     prob_data.T0.repeat(args.num_samples[0],1,1),
                #     prob_data.num_joints.expand(args.num_samples[0]),
                #     T_final = T_final.expand(args.num_samples[0],-1,-1)
                # )

                # T = batchFKmultiDOF(
                #     prob_data.T0.repeat(args.num_samples[0],1,1),
                #     q_sols_.reshape(-1,1),
                #     prob_data.num_joints.expand(args.num_samples[0])
                # )
                # ee_ind = torch.cumsum(prob_data.num_joints.expand(args.num_samples[0]) + 1, dim=0) - 1 # end indices of joints

                q_sols_ = batchIKmultiDOF(
                    P_all.reshape(-1,3),
                    prob_data.T0.repeat(num_samples_pre,1,1),
                    prob_data.num_joints.expand(num_samples_pre),
                    T_final = T_final.expand(num_samples_pre,-1,-1)
                ).reshape(-1,prob_data.num_joints)

                T = batchFKmultiDOF(
                    prob_data.T0.repeat(num_samples_pre,1,1),
                    q_sols_.reshape(-1,1),
                    prob_data.num_joints.expand(num_samples_pre)
                )
                T_ee = T[ee_ind]
                ind = filter_by_error(T_ee, T_final_inv.expand(num_samples_pre,-1,-1))
                t_sol = time.time() - t0

                q_sols_ = q_sols_[ind[:args.num_samples[0]]]
                e_pose = torch_log_from_T(torch.bmm(T_final_inv.expand(args.num_samples[0],-1,-1), T_ee[ind[:args.num_samples[0]]]))
                e_pose_norm = torch.norm(e_pose, dim=-1).cpu().numpy()
                e_pos_norm = torch.norm(e_pose[:,:3], dim=-1).cpu().numpy()
                e_rot_norm = torch.norm(e_pose[:,3:], dim=-1).cpu().numpy()
                for idx in range(args.num_samples[0]):
                    entry = {
                        "Id": kdx,
                        "Robot": robot_type,
                        "Goal Pose": T_goal,
                        "Sol. Config": q_sols_[idx],
                        # "Sol. Points": P_all[idx,:],
                        "Err. Pose": e_pose_norm[idx],
                        "Err. Position": e_pos_norm[idx],
                        "Err. Rotation": e_rot_norm[idx],
                        # "Goal Config": q_goal_np,
                        # "Goal Points": P_goal,
                        "Sol. Time": t_sol,
                    }
                    sol_data.append(entry)
                all_sol_data.append(pd.DataFrame(sol_data))
                print(kdx)

        pd_data = pd.concat(all_sol_data)

        # exp_dir = f"{sys.path[0]}/results/"+ f"{args.id}/"
        exp_dir = f"{sys.path[0]}/results/TRO/"+ f"{args.id}/"
        os.makedirs(exp_dir, exist_ok=True)
        pd_data.to_pickle(os.path.join(exp_dir, "results.pkl"))
        pd_data.to_csv(os.path.join(exp_dir, "results.csv"))


if __name__ == "__main__":
    random.seed(17)
    args = parse_analysis_args()
    main(args)
