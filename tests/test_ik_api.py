import itertools
from typing import Tuple
import unittest

import numpy as np
import torch

from generative_graphik.utils.dataset_generation import random_revolute_robot_graph
from generative_graphik.utils.api import ik
from generative_graphik.utils.get_model import get_model
from generative_graphik.utils.torch_to_graphik import joint_transforms_from_t_zeros, joint_transforms_to_t_zero


class ApiTests(unittest.TestCase):
    """
    Tests the generative_graphik.utils.api functionalities.
    """

    def setUp(self):
        try:
            self.model = get_model()
        except FileNotFoundError as exe:
            print(exe)
            if exe.filename.split('/')[-1] == 'config.yaml':
                raise FileNotFoundError("No configuration file found. Create a config.yaml file similar to "
                                        "sample_config.yaml and place it in the root of the project.")
            else:
                raise FileNotFoundError("No model found. Train a model and place it in the directory specified in the "
                                        "config.yaml file.")

    @staticmethod
    def ik_error(T_desired: torch.Tensor, T_eef: torch.Tensor) -> Tuple[float, float]:
        """Utility function to compute the error between two SE3 matrices."""
        r_desired = T_desired[:3, :3]
        r_eef = T_eef[:3, :3]
        t_desired = T_desired[:3, 3]
        t_eef = T_eef[:3, 3]
        e_rot = torch.arccos((torch.trace(r_desired.T @ r_eef) - 1) / 2) * 180 / np.pi  # degrees
        e_trans = torch.norm(t_desired - t_eef)  # meters
        return e_rot.item(), e_trans.item()

    def test_conversions(self, N=10, dof=6):
        """Test that the joint transforms (torch) can be converted to T_zero (dict of SE3) and back."""
        for _ in range(N):
            g = random_revolute_robot_graph(dof)
            T_zero = g.robot.from_dh_params(g.robot.params)
            transforms = joint_transforms_from_t_zeros(T_zero, keys=g.robot.joint_ids)
            T_zero_reconstructed = joint_transforms_to_t_zero(transforms, keys=g.robot.joint_ids)
            for key in T_zero:
                self.assertTrue(np.allclose(T_zero[key].as_matrix(), T_zero_reconstructed[key].as_matrix()))

    def test_ik_api(self, nR: int = 8, nG: int = 8, samples: int = 8, dof: int = 6):
        """
        Test the inverse kinematics API, i.e., an inverse kinematics functionality that is framework-agnostic and does
        not require the user to know the details of the generative_graphik approach.
        """
        graphs = [random_revolute_robot_graph(dof) for _ in range(nR)]
        goals = dict.fromkeys(range(nR), None)
        for i, j in itertools.product(range(nR), range(nG)):
            if j == 0:
                goals[i] = []
            q = torch.rand(dof + 1) * 2 * torch.pi - torch.pi
            # q[-1] = 0
            angles = {jnt: q_jnt.item() for jnt, q_jnt in zip(graphs[i].robot.joint_ids, q)}
            T = graphs[i].robot.pose(angles, graphs[i].robot.end_effectors[-1])
            goals[i].append(torch.Tensor(T.as_matrix()))

        for i, g in enumerate(graphs):
            T_zero_native = g.robot.from_dh_params(g.robot.params)
            transforms = joint_transforms_from_t_zeros(T_zero_native, keys=g.robot.joint_ids)
            transforms = torch.unsqueeze(transforms, 0)  # T=D=
            sol = ik(transforms, torch.stack(goals[i]), samples=samples, return_all=True)

            trans_errors = list()
            rot_errors = list()

            for k, l in itertools.product(range(nG), range(samples)):
                q_kl = {jnt: sol[0, k, l, m].item() for m, jnt in enumerate(g.robot.joint_ids[1:])}
                q_kl['p0'] = 0
                T = g.robot.pose(q_kl, g.robot.end_effectors[-1]).as_matrix()
                e_rot, e_trans = self.ik_error(goals[i][k], torch.Tensor(T))
                rot_errors.append(e_rot)
                trans_errors.append(e_trans)

            # Get at least one good solution
            self.assertLessEqual(np.min(rot_errors), 2)
            self.assertLessEqual(np.min(trans_errors), 0.05)

            # Is it significantly better than random? (educated guess of what a random precision would be)
            self.assertLessEqual(np.mean(rot_errors), 45)
            self.assertLessEqual(np.min(trans_errors), np.mean([np.linalg.norm(goals[i][j][:3, 3]) for j in range(nG)]) / 10)


if __name__ == '__main__':
    unittest.main()
