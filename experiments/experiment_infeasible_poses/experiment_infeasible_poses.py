import importlib.util
import argparse
import random
import os
from argparse import Namespace
import json
import pickle
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset

import graphik
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
    dataset_paths = args.dataset_paths

    # Initialize tracIK
    for robot_type, model_path, infeasible_pose_path, dataset_path in zip(robot_types, model_paths, infeasible_pose_paths, dataset_paths):
        
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

        # Load dataset to compare to
        with open(os.path.join(dataset_path, "np_poses.pkl"), 'rb') as f:
            dataset_poses = pickle.load(f)

        print(infeasible_poses.shape)
        print(dataset_poses.shape)
        assert 0
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
        
        print(type(infeasible_poses))
        print(type(reference_dataset))
        # prob_data = generate_data_point_from_pose(graph).to(args.device)
        # prob_data.num_graphs = 1

def parse_experiment_infeasible_poses_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--model_paths", nargs="*", type=str, default=["/home/olimoyo/generative-graphik/saved_models/paper_models/kuka_512k_model"], help="Path to folder with model")
    parser.add_argument("--infeasible_pose_paths", nargs="*", type=str, default="/home/olimoyo/generative-graphik/datasets/infeasible_poses/infeasible_poses_kuka.pkl", help="Path to folder with infeasible poses to test with.")
    parser.add_argument("--robots", nargs="*", type=str, default=["kuka"], help="Robots to test on")
    parser.add_argument("--dataset_paths", nargs="*", type=str, default="/media/stonehenge/users/oliver-limoyo/2.56m-kuka", help="Path to folder with infeasible poses to test with.")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(3)
    args = parse_experiment_infeasible_poses_args()
    infeasible_poses = run_experiment_infeasible_poses(args)

