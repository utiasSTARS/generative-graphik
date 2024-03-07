#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 06.03.24
import itertools
from typing import Dict, Sequence

from liegroups.numpy.se3 import SE3Matrix
import torch
from torch_geometric.data import Data

from generative_graphik.utils.dataset_generation import StructData


def define_ik_data(robot_data: StructData, goals: torch.Tensor) -> Data:
    """
    This function takes a robot and a set of goals and returns a data point for every goal.

    :param robot_data: A StructData object containing the robot's kinematics.
    :param goals: A tensor of shape (nG, 4, 4) containing the desired end-effector poses.
    """
    pass


def joint_transforms_from_t_zeros(T_zero: Dict[str, SE3Matrix], keys: Sequence[str]) -> torch.Tensor:
    """Assumes that joints are alphabetically sorted"""
    ret = torch.zeros((len(T_zero) - 1, 4, 4))
    for i in range(1, len(keys)):
        ret[i - 1] = torch.Tensor(T_zero[keys[i-1]].inv().dot(T_zero[keys[i]]).as_matrix())
    return ret


def joint_transforms_to_t_zero(transforms: torch.Tensor, keys: Sequence[str]) -> Dict[str, SE3Matrix]:
    """
    This function takes a tensor of joint transformations and returns the t_zero tensor, which describes the joint
    pose in the world frame for the zero configuration.

    :param transforms: A tensor of shape (nJ, 4, 4).
    :param keys: The keys to use for the joint names. Assumes the first key is for the world frame, thus it will be
        set to the identity.
    """
    nj = transforms.shape[0]
    t_zero = transforms.clone()
    for i in range(1, nj):
        t_zero[i] = t_zero[i - 1] @ t_zero[i]
    t_zero = {keys[i+1]: SE3Matrix.from_matrix(t_zero[i].detach().cpu().numpy()) for i in range(nj)}
    t_zero[keys[0]] = SE3Matrix.identity()
    return t_zero
