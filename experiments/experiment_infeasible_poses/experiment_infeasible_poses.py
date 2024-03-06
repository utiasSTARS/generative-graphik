import importlib.util
import argparse
import random
import os
from argparse import Namespace
import json
import pickle
import numpy as np
import sys

import torch
from torch_geometric.data import InMemoryDataset
from liegroups.numpy import SE3
import pandas as pd

import graphik
from graphik.utils.dgp import graph_from_pos
from graphik.utils.roboturdf import (
    RobotURDF,
    load_ur10,
    load_kuka,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_panda,
)
from generative_graphik.utils.dataset_generation import generate_data_point_from_pose

def model_arg_loader(path):
    """Load hyperparameters from trained model."""
    if os.path.isdir(path):
        with open(os.path.join(path, "hyperparameters.txt"), "r") as fp:
            return Namespace(**json.load(fp))
        
class CachedDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super(CachedDataset, self).__init__(None)
        self.data, self.slices = data, slices

def run_experiment_infeasible_poses(args):
    robot_types = args.robots
    model_paths = args.model_paths
    infeasible_pose_paths = args.infeasible_pose_paths

    all_sol_data = []
    # Initialize tracIK
    for robot_type, model_path, infeasible_pose_path in zip(robot_types, model_paths, infeasible_pose_paths):
        
        # Load model
        spec = importlib.util.spec_from_file_location("model", os.path.join(model_path, "model.py"))
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        model_args = model_arg_loader(model_path)
        model = model.Model(model_args).to(args.device)

        # Load infeasible poses
        with open(infeasible_pose_path, 'rb') as f:
            infeasible_poses_list = pickle.load(f)

        infeasible_poses = []
        for infeasible_pose in infeasible_poses_list:
            infeasible_pose = infeasible_pose[None, ...]
            infeasible_poses.append(infeasible_pose)
        infeasible_poses = np.concatenate(infeasible_poses)

        # Load problem
        if robot_type == "ur10":
            robot, graph = load_ur10(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
            link_base, link_ee = 'base_link', 'ee_link'
        elif robot_type == "kuka":
            robot, graph = load_kuka(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
            link_base, link_ee = 'lbr_iiwa_link_0', 'ee_link'
        elif robot_type == "lwa4d":
            robot, graph = load_schunk_lwa4d(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
            link_base, link_ee = 'lwa4d_base_link', 'lwa4d_ee_link'
        elif robot_type == "panda":
            robot, graph = load_panda(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
            link_base, link_ee = 'panda_link0', 'panda_link7'
        elif robot_type == "lwa4p":
            robot, graph = load_schunk_lwa4p(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
            link_base, link_ee = 'lwa4p_base_link', 'lwa4p_ee_link'
        else:
            raise NotImplementedError
        
        sol_data = []        
        for kdx, infeasible_pose in enumerate(infeasible_poses):
            print(robot_type, f"{kdx + 1} / {len(infeasible_poses)}")
            T_goal = SE3.from_matrix(infeasible_pose)
            prob_data = generate_data_point_from_pose(graph, T_goal).to(args.device)
            prob_data.num_graphs = 1
            data = model.preprocess(prob_data)
            num_samples_pre = args.num_samples * 4

            # Compute solutions
            P_all = (
                    model.forward_eval(
                        x=data.pos, 
                        h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1), 
                        edge_attr=data.edge_attr, 
                        edge_attr_partial=data.edge_attr_partial, 
                        edge_index=data.edge_index_full, 
                        partial_goal_mask=data.partial_goal_mask, 
                        nodes_per_single_graph= int(data.num_nodes / 1),
                        batch_size=1,
                        num_samples=num_samples_pre
                    )
            ).cpu().detach().numpy()

            # Analyze solutions
            e_pose = np.empty([P_all.shape[0]])
            e_pos = np.empty([P_all.shape[0]])
            e_rot = np.empty([P_all.shape[0]])
            q_sols_np = np.empty([P_all.shape[0], robot.n])
            for idx in range(P_all.shape[0]):
                P = P_all[idx, :]

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
            idx_sorted = np.argsort(e_pose)

            for ii in idx_sorted[:args.num_samples]:
                entry = {
                    "Id": kdx,
                    "Robot": robot_type,
                    "Goal Pose": T_goal.as_matrix(),
                    "Sol. Config": q_sols_np[ii],
                    "Err. Pose": e_pose[ii],
                    "Err. Position": e_pos[ii],
                    "Err. Rotation": e_rot[ii],
                }
                sol_data.append(entry)
            all_sol_data.append(pd.DataFrame(sol_data))

    pd_data = pd.concat(all_sol_data)
    exp_dir = f"{sys.path[0]}/results/"
    os.makedirs(exp_dir, exist_ok=True)
    pd_data.to_pickle(os.path.join(exp_dir, "results.pkl"))

def parse_experiment_infeasible_poses_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--model_paths", nargs="*", type=str, default=["/home/olimoyo/generative-graphik/saved_models/paper_models/kuka_512k_model"], help="Path to folder with model")
    parser.add_argument("--infeasible_pose_paths", nargs="*", type=str, default=["/home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_kuka.pkl"], help="Path to folder with infeasible poses to test with.")
    parser.add_argument("--robots", nargs="*", type=str, default=["kuka"], help="Robots to test on")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument("--num_samples", type=int, default=32, help="Total number of samples per problem")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(3)
    args = parse_experiment_infeasible_poses_args()
    infeasible_poses = run_experiment_infeasible_poses(args)

