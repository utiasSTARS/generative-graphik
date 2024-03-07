import itertools
from typing import Callable

from liegroups.numpy.se3 import SE3Matrix
import torch

from graphik.graphs import ProblemGraphRevolute
from graphik.robots import RobotRevolute
from graphik.utils import graph_from_pos
from generative_graphik.utils.dataset_generation import generate_data_point_from_pose
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


def ik(kinematic_chains: torch.tensor,
       goals: torch.tensor,
       samples: int = 16,
       return_all: bool = False,
       ik_cost_function: Callable = _default_cost_function) -> torch.Tensor:
    """
    This function takes robot kinematics and any number of goals and solves the inverse kinematics, using graphIK.

    :param kinematic_chains: A tensor of shape (nR, N, 4, 4) containing the joint transformations of nR robots with N
        joints each.
    :param goals: A tensor of shape (nR, nG, 4, 4) containing the desired end-effector poses.
    :param samples: The number of samples to use for the forward pass of the model.
    :param return_all: If True, returns all the samples from the forward pass, so the resulting tensor has a shape
        nR x nG x samples x nJ. If False, returns the best one only, so the resulting tensor has a shape nR x nG x nJ.
    :param ik_cost_function: The cost function to use for the inverse kinematics problem if return_all is False.
    """
    device = kinematic_chains.device
    model = get_model().to(device)

    assert len(kinematic_chains.shape) == 4, f'Expected 4D tensor, got {kinematic_chains.shape}'
    nr, nj, _, _ = kinematic_chains.shape
    _, nG, _, _ = goals.shape

    t_zeros = {i: joint_transforms_to_t_zero(kinematic_chains[i], [f'p{j}' for j in range(1 + nj)], se3type='numpy') for i in range(nr)}
    robots = {i: RobotRevolute({'num_joints': nj, 'T_zero': t_zeros[i]}) for i in range(nr)}
    graphs = {i: ProblemGraphRevolute(robots[i]) for i in range(nr)}
    if return_all:
        q = torch.zeros((nr, nG, samples, nj), device=device)
    else:
        q = torch.zeros((nr, nG, nj), device=device)

    for i, j in itertools.product(range(nr), range(nG)):
        graph = graphs[i]
        robot = robots[i]
        goal = goals[i, j]
        problem = generate_data_point_from_pose(graph, goal).to(device)
        problem = model.preprocess(problem)
        P_all = (
            model.forward_eval(
                x=problem.pos,
                h=torch.cat((problem.type, problem.goal_data_repeated_per_node), dim=-1),
                edge_attr=problem.edge_attr,
                edge_attr_partial=problem.edge_attr_partial,
                edge_index=problem.edge_index_full,
                partial_goal_mask=problem.partial_goal_mask,
                nodes_per_single_graph=int(problem.num_nodes / 1),
                batch_size=1,
                num_samples=samples
            )
        )
        best = float('inf')
        for k, p_k in enumerate(P_all):
            q_k = graph.joint_variables(graph_from_pos(p_k.detach().cpu().numpy(), graph.node_ids),
                                        {robot.end_effectors[0]: SE3Matrix.from_matrix(goal.detach().cpu().numpy(),
                                                                                       normalize=True)})
            if return_all:
                q[i, j, k] = torch.tensor([q_k[key] for key in robot.joint_ids[1:]], device=device)
                continue
            T_ee = graph.robot.pose(q_k, robot.end_effectors[-1])
            cost = ik_cost_function(goal, torch.tensor(T_ee.as_matrix()).to(goal))
            if cost < best:
                best = cost
                q[i, j] = torch.tensor([q_k[key] for key in robot.joint_ids[1:]], device=device)
    return q
