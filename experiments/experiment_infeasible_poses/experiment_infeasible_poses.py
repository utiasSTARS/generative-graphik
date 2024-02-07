import argparse
import random

import torch
import numpy as np

from generative_graphik.utils.dataset_generation import CachedDataset
from generative_graphik.utils.dataset_generation import generate_data_point
from tracikpy import TracIKSolver
import graphik
from graphik.utils.roboturdf import (
    RobotURDF,
    load_ur10,
    load_kuka,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_panda,
)

def ee_error(ee1, ee2):
    ee_diff = np.linalg.inv(ee1) @ ee2
    trans_err = np.linalg.norm(ee_diff[:3, 3], ord=1)
    angle_err = np.arccos(np.trace(ee_diff[:3, :3] - 1) / 2)
    return trans_err, angle_err

def find_infeasible_poses(args):
    robot_types = args.robots
    device = args.device

    # Initialize tracIK
    for robot_type in robot_types:
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
        
    ik_solver = TracIKSolver(fname, link_base, link_ee, timeout=0.5)

    # Pick the top N as seed poses and add random noise to find infeasible poses
    unfeasible_poses = []
    for _ in range(1000):

        # Generate random problem
        prob_data = generate_data_point(graph).to(device)
        q_goal = prob_data.q_goal.numpy()
        q_goal[-1] += 0.0001
        T_goal = ik_solver.fk(q_goal)

        # 32 random retries
        for _ in range(32):
            q_init = np.array(list(robot.random_configuration().values()))
            # Solve using tracIK and find poses that fail
            qout = ik_solver.ik(
                T_goal, 
                qinit=q_init,
                bx = 1e-2, 
                by = 1e-2, 
                bz = 1e-2
            )
            if qout is not None:
                break

        if qout is None:
            print("None detected")

def parse_experiment_infeasible_poses_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    # parser.add_argument("--model_path", type=str, required=True, help="Path to folder with model data")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument("--robots", nargs="*", type=str, default=["panda"], help="Robots to test on")
    parser.add_argument("--n_infeasible_poses", type=int, default=1000, help="Number of infeasible poses to test on")
    parser.add_argument("--translational_noise", type=float, default=0.2, help="Noise in metres to add.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(3)
    args = parse_experiment_infeasible_poses_args()
    infeasible_poses = find_infeasible_poses(args)
