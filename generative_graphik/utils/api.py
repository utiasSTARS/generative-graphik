import itertools
from typing import Callable, Optional

from liegroups.numpy.se3 import SE3Matrix
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from graphik.graphs import ProblemGraphRevolute
from graphik.robots import RobotRevolute
from graphik.utils import graph_from_pos
from generative_graphik.utils.dataset_generation import generate_data_point_from_pose, create_dataset_from_data_points
from generative_graphik.utils.get_model import get_model
from generative_graphik.utils.torch_to_graphik import joint_transforms_to_t_zero


def _default_cost_function(T_desired: torch.Tensor, T_eef: torch.Tensor) -> torch.Tensor:
    """
    The default cost function for the inverse kinematics problem. It is the sum of the squared errors between the
    desired and actual end-effector poses.

    :param T_desired: The desired end-effector pose.
    :param T_eef: The actual end-effector pose.
    :return: The cost.
    """
    return torch.sum((T_desired - T_eef) ** 2)


def _get_goal_idx(num_robots, samples_per_robot, batch_size, num_batch, idx_batch):
    num_sample = num_batch * batch_size + idx_batch
    return num_sample % samples_per_robot

def _get_robot_idx(num_robots, samples_per_robot, batch_size, num_batch, idx_batch):
    num_sample = num_batch * batch_size + idx_batch
    return num_sample // samples_per_robot


def ik(kinematic_chains: torch.tensor,
       goals: torch.tensor,
       samples: int = 16,
       return_all: bool = False,
       ik_cost_function: Callable = _default_cost_function,
       batch_size: int = 64,
       ) -> torch.Tensor:
    """
    This function takes robot kinematics and any number of goals and solves the inverse kinematics, using graphIK.

    :param kinematic_chains: A tensor of shape (nR, N, 4, 4) containing the joint transformations of nR robots with N
        joints each.
    :param goals: A tensor of shape (nR, nG, 4, 4) containing the desired end-effector poses.
    :param samples: The number of samples to use for the forward pass of the model.
    :param return_all: If True, returns all the samples from the forward pass, so the resulting tensor has a shape
        nR x nG x samples x nJ. If False, returns the best one only, so the resulting tensor has a shape nR x nG x nJ.
    :param ik_cost_function: The cost function to use for the inverse kinematics problem if return_all is False.
    :return: See return_all for info.
    """
    device = kinematic_chains.device
    model = get_model().to(device)

    assert len(kinematic_chains.shape) == 4, f'Expected 4D tensor, got {kinematic_chains.shape}'
    nr, nj, _, _ = kinematic_chains.shape
    _, nG, _, _ = goals.shape
    eef = f'p{nj}'

    t_zeros = {i: joint_transforms_to_t_zero(kinematic_chains[i], [f'p{j}' for j in range(1 + nj)], se3type='numpy') for
               i in range(nr)}
    robots = {i: RobotRevolute({'num_joints': nj, 'T_zero': t_zeros[i]}) for i in range(nr)}
    graphs = {i: ProblemGraphRevolute(robots[i]) for i in range(nr)}
    if return_all:
        q = torch.zeros((nr, nG, samples, nj), device=device)
    else:
        q = torch.zeros((nr, nG, nj), device=device)

    problems = list()
    for i, j in itertools.product(range(nr), range(nG)):
        graph = graphs[i]
        goal = goals[i, j]
        problems.append(generate_data_point_from_pose(graph, goal))

    # FIXME: Create one data point per sample until forward_eval works correctly with more than one sample
    problems_times_samples = list(itertools.chain.from_iterable(zip(*[problems] * samples)))
    data = create_dataset_from_data_points(problems_times_samples)
    batch_size_forward = batch_size * samples
    loader = DataLoader(data, batch_size=batch_size_forward, shuffle=False, num_workers=0)

    for i, problem in enumerate(loader):
        problem = model.preprocess(problem)
        b = len(problem)  # The actual batch size (might be smaller than batch_size_forward at the end of the dataset)
        num_nodes_per_graph = int(problem.num_nodes / b)
        P_all_ = model.forward_eval(
            x=problem.pos,
            h=torch.cat((problem.type, problem.goal_data_repeated_per_node), dim=-1),
            edge_attr=problem.edge_attr,
            edge_attr_partial=problem.edge_attr_partial,
            edge_index=problem.edge_index_full,
            partial_goal_mask=problem.partial_goal_mask,
            nodes_per_single_graph=num_nodes_per_graph,
            batch_size=b,
            num_samples=1
        ).squeeze()
        # Rearrange, s.t. we have problem_nr x sample_nr x node_nr x 3
        P_all = P_all_.view(b // samples, samples, num_nodes_per_graph, 3)

        for idx in range(b // samples):
            idx_robot = _get_robot_idx(nr, nG, batch_size, i, idx)
            idx_goal = _get_goal_idx(nr, nG, batch_size, i, idx)
            graph = graphs[idx_robot]
            goal = goals[idx_robot, idx_goal]
            goalse3 = SE3Matrix.from_matrix(goal.detach().cpu().numpy(), normalize=True)
            best = float('inf')
            for sample in range(samples):
                P = P_all[idx, sample, ...]
                q_s = graph.joint_variables(graph_from_pos(P.detach().cpu().numpy(), graph.node_ids), {eef: goalse3})
                if return_all:
                    q[idx_robot, idx_goal, sample] = torch.tensor([q_s[key] for key in robots[idx_robot].joint_ids[1:]], device=device)
                T_ee = robots[idx_robot].pose(q_s, eef)
                cost = ik_cost_function(goal, torch.tensor(T_ee.as_matrix()).to(goal))
                if cost < best:
                    best = cost
                    q[idx_robot, idx_goal] = torch.tensor([q_s[key] for key in robots[idx_robot].joint_ids[1:]], device=device)
    return q
