import importlib.util
import json
import os
import sys

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
from generative_graphik.utils.dataset_generation import (
    generate_data_point,
    random_revolute_robot_graph,
)
from graphik.utils.roboturdf import (
    RobotURDF,
    load_ur10,
    load_kuka,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_panda,
)
import pyrender

from graphik.utils.dgp import graph_from_pos
from liegroups.numpy import SE3, SO3
from graphik.utils.urdf_visualization import make_scene


def model_arg_loader(path):
    """Load hyperparameters from trained model."""
    if os.path.isdir(path):
        with open(os.path.join(path, "hyperparameters.txt"), "r") as fp:
            return Namespace(**json.load(fp))


def plot_revolute_manipulator_robot(robot, configs, transparency=0.7):

    # scene = make_scene(robot, with_balls=False, with_edges=False)
    scene = None
    for idx in range(len(configs)):
        if idx==0:
            trn=1
        else:
            trn=transparency
        scene = make_scene(
            robot,
            scene = scene,
            q=configs[idx],
            with_frames=False,
            with_balls=False,
            with_edges=False,
            with_robot=True,
            transparency=trn,
        )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2) / 2

    # panda
    # camera_pose = SE3(SO3.identity(), np.array([0,-1.45,0.25])).dot(SE3(SO3.rotx(np.pi/2), np.array([0,0,0])))
    # ur10
    # camera_pose = SE3(SO3.identity(), np.array([0,-1.75,0.35])).dot(SE3(SO3.rotx(np.pi/2), np.array([0,0,0])))
    # kuka
    # camera_pose = SE3(SO3.identity(), np.array([0,-1.75,0.35])).dot(SE3(SO3.rotx(np.pi/2), np.array([0,0,0])))
    # lwa4d
    camera_pose = SE3(SO3.identity(), np.array([0,-1.75,0.35])).dot(SE3(SO3.rotx(np.pi/2), np.array([0,0,0])))
    camera_pose = camera_pose.as_matrix()
    # camera_pose = np.array(
    #     [
    #         [0.0, -s, s, 1.25],
    #         [1.0, 0.0, 0.0, 0.0],
    #         [0.0, s, s, 1.5],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=3.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(2048, 2048)
    color, depth = r.render(scene)
    return color


# NOTE generates all the initializations and stores them to a pickle file
def main(args):
    device = args.device
    num_evals = args.n_evals  # number of evaluations
    robot_types = args.robots
    dofs = args.dofs  # number of dof we test on

    evals_per_robot = num_evals // len(robot_types)  # number of evaluations per robots
    spec = importlib.util.spec_from_file_location("model", args.model_path[0] + "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    # load models
    model_args = model_arg_loader(args.model_path[0])
    model = model.Model(model_args).to(device)
    name = model_args.id.replace("model", "results")
    exp_dir = f"{sys.path[0]}/results/"+ f"{args.id}/images/"
    os.makedirs(exp_dir, exist_ok=True)
    c = np.pi / 180

    if args.model_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.model_path[0] + f"/net.pth", map_location=device)
            )
            model.eval()
        except Exception as e:
            print(e)

    all_sol_data = []
    # fig_handle, ax_handle = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    for robot_type in robot_types:
        if robot_type == "ur10":
            limits_l = -np.array([180, 180, 180, 180, 180, 180]) * c
            limits_u = np.array([180, 180, 180, 180, 180, 180]) * c
            limits = [limits_l, limits_u]
            robot, graph = load_ur10(limits=limits)
            fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
            urdf_robot = RobotURDF(fname)
        elif robot_type == "kuka":
            limits_l = -np.array([170, 120, 170, 120, 170, 120, 170]) * c
            limits_u = np.array([170, 120, 170, 120, 170, 120, 170]) * c
            limits = [limits_l, limits_u]
            robot, graph = load_kuka(limits=limits)
            fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
            urdf_robot = RobotURDF(fname)
        elif robot_type == "lwa4d":
            limits_l = -np.array([180, 123, 180, 125, 180, 170, 170]) * c
            limits_u = np.array([180, 123, 180, 125, 180, 170, 170]) * c
            limits = [limits_l, limits_u]
            robot, graph = load_schunk_lwa4d(limits=None)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
            urdf_robot = RobotURDF(fname)
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
        elif robot_type == "lwa4p":
            limits_l = -np.array([170, 170, 155.3, 170, 170, 170]) * c
            limits_u = np.array([170, 170, 155.3, 170, 170, 170]) * c
            limits = [limits_l, limits_u]
            robot, graph = load_schunk_lwa4p(limits=limits)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
            urdf_robot = RobotURDF(fname)
        else:
            raise NotImplementedError
        fig= plt.figure()
        for kdx in range(evals_per_robot):
            sol_data = []

            # Generate random problem
            prob_data = generate_data_point(graph).to(device)
            prob_data.num_graphs = 1
            data = model.preprocess(prob_data)
            P_goal = data.pos.cpu().numpy()
            T_goal = SE3.exp(data.T_ee.cpu().numpy())
            q_goal = graph.joint_variables(graph_from_pos(P_goal, graph.node_ids))

            # Compute solutions
            P_all = (
                model.forward_eval(data, num_samples=args.num_samples[0]*4)
                .cpu()
                .detach()
                # .numpy()
            )
            src, dst = data["edge_index_full"]
            dist_samples = ((P_all[:,src] - P_all[:,dst])**2).sum(dim=-1).sqrt()
            dist_diff = dist_samples[:, data.partial_mask] - data["edge_attr_partial"].t()
            dist_diff_norm, _ = torch.max(torch.abs(dist_diff), dim=-1)
            ind = torch.argsort(dist_diff_norm)
            P_all = P_all[ind[:args.num_samples[0]]]

            # torch.cuda.synchronize()

            # Analyze solutions
            q_sols = [q_goal]
            for idx in range(P_all.shape[0]):
                P = P_all[idx, :]

                q_sol = graph.joint_variables(
                    graph_from_pos(P, graph.node_ids), {robot.end_effectors[0]: T_goal}
                )  # get joint angles
                q_sols.append(q_sol)

            config_img = plot_revolute_manipulator_robot(
                urdf_robot,
                q_sols,
                transparency=0.15
            )
            fig = plt.imshow(config_img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # fig.axes.set_title("Distribution")
            # plt.pause(0.1)
            os.makedirs(exp_dir, exist_ok=True)
            plt.savefig(exp_dir + str(robot_type) + '_' + str(kdx) + '.png', dpi=512)


if __name__ == "__main__":
    random.seed(17)
    args = parse_analysis_args()
    main(args)
