import importlib.util
import json
import os
import sys
import tracikpy
import random
import copy
import pandas as pd
import time
import graphik
import numpy as np
import torch

from argparse import Namespace
from tracikpy import TracIKSolver
from graphik.graphs import ProblemGraphRevolute
from graphik.graphs.graph_revolute import list_to_variable_dict
from graphik.robots import RobotRevolute
from torch_geometric.data import DataLoader
from generative_graphik.utils.dataset_generation import generate_data_point
# from generative_graphik.utils.torch_utils import SE3_inv, batchFKmultiDOF, batchIKmultiDOF, node_attributes, torch_log_from_T

os.environ["PYOPENGL_PLATFORM"] = "egl"

from generative_graphik.args.parser import parse_analysis_args
from graphik.utils.roboturdf import (
    RobotURDF,
    load_ur10,
    load_kuka,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_panda,
)

from liegroups.numpy import SE3, SO3

# NOTE generates all the initializations and stores them to a pickle file
def main(args):
    np.random.seed(0)
    device = args.device
    num_evals = args.n_evals  # number of evaluations
    robot_types = args.robots
    dofs = args.dofs  # number of dof we test on
    evals_per_robot = num_evals // len(robot_types)
    all_sol_data = []
    for robot_type in robot_types:
        if robot_type == "ur10":
            robot, graph = load_ur10(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
            link_base, link_ee = 'base_link', 'ee_link'
        elif robot_type == "kuka":
            # limits_l = -np.array([170, 120, 170, 120, 170, 120, 170]) * c
            # limits_u = np.array([170, 120, 170, 120, 170, 120, 170]) * c
            # limits = [limits_l, limits_u]
            robot, graph = load_kuka(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
            link_base, link_ee = 'lbr_iiwa_link_0', 'ee_link'
        elif robot_type == "lwa4d":
            # limits_l = -np.array([180, 123, 180, 125, 180, 170, 170]) * c
            # limits_u = np.array([180, 123, 180, 125, 180, 170, 170]) * c
            # limits = [limits_l, limits_u]
            robot, graph = load_schunk_lwa4d(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
            link_base, link_ee = 'lwa4d_base_link', 'lwa4d_ee_link'
        elif robot_type == "panda":
            # limits_l = -np.array(
            #     [2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 0.0175, 2.8973]
            # )
            # limits_u = np.array(
            #     [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
            # )
            # limits = [limits_l, limits_u]
            robot, graph = load_panda(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
            link_base, link_ee = 'panda_link0', 'panda_link7'
        elif robot_type == "lwa4p":
            # limits_l = -np.array([170, 170, 155.3, 170, 170, 170]) * c
            # limits_u = np.array([170, 170, 155.3, 170, 170, 170]) * c
            # limits = [limits_l, limits_u]
            robot, graph = load_schunk_lwa4p(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
            link_base, link_ee = 'lwa4p_base_link', 'lwa4p_ee_link'
        else:
            raise NotImplementedError

        ik_solver = TracIKSolver(fname, link_base, link_ee, timeout=0.1)
        ub = np.ones(ik_solver.number_of_joints) * np.pi
        lb = -ub
        ik_solver.joint_limits = lb, ub
        q_init = np.zeros(ik_solver.number_of_joints)
        for kdx in range(evals_per_robot):
            sol_data = []
            # q_init = np.array(list(robot.random_configuration().values()))

            # Generate random problem
            prob_data = generate_data_point(graph).to(device)
            q_goal = prob_data.q_goal.numpy()
            q_goal[-1] += 0.0001
            T_goal = ik_solver.fk(q_goal)

            t0 = time.time()
            q_sol = ik_solver.ik(T_goal, qinit=q_init, bx = 1e-3, by = 1e-3, bz = 1e-3)
            t_sol = time.time() - t0
            if q_sol is None:
                import ipdb; ipdb.set_trace()

            T_ee = ik_solver.fk(q_sol)

            e_pose = (SE3.from_matrix(T_goal, normalize=True).inv().dot(SE3.from_matrix(T_ee, normalize=True))).log()
            e_pose_norm = np.linalg.norm(e_pose)
            e_pos_norm = np.linalg.norm(e_pose[:3])
            e_rot_norm = np.linalg.norm(e_pose[3:])

            entry = {
                "Id": kdx,
                "Robot": robot_type,
                "Goal Pose": SE3.from_matrix(T_goal),
                "Sol. Config": [q_sol],
                # "Sol. Points": P_all[idx,:],
                "Err. Pose": e_pose_norm,
                "Err. Position": e_pos_norm,
                "Err. Rotation": e_rot_norm,
                # "Goal Config": q_goal_np,
                # "Goal Points": P_goal,
                "Sol. Time": t_sol,
            }
            all_sol_data.append(pd.DataFrame(entry))
            # print(kdx)

    pd_data = pd.concat(all_sol_data)
    import ipdb; ipdb.set_trace()

    for robot_type in robot_types:
        print(robot_type)
        success_data = pd_data[(pd_data['Err. Position'] < 1e-2) & (pd_data['Err. Rotation'] < (180/np.pi)) & (pd_data['Robot']==robot_type)]
        print(success_data.describe())
        print(len(success_data)/evals_per_robot)
        print('------------------------')
    # exp_dir = f"{sys.path[0]}/results/TRO/"+ f"{args.id}/"
    # os.makedirs(exp_dir, exist_ok=True)
    # pd_data.to_pickle(os.path.join(exp_dir, "results.pkl"))
    # pd_data.to_csv(os.path.join(exp_dir, "results.csv"))


if __name__ == "__main__":
    random.seed(17)
    args = parse_analysis_args()
    main(args)
