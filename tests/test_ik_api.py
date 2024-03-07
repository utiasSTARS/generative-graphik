import itertools
from time import time
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
        """
        These tests rely on a trained model that needs to be present.
        Its location should be specified in the config.yaml file.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = get_model().to(self.device)
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
            transforms = joint_transforms_from_t_zeros(T_zero, keys=g.robot.joint_ids, device=self.device)
            T_zero_reconstructed = joint_transforms_to_t_zero(transforms, keys=g.robot.joint_ids)
            for key in T_zero:
                self.assertTrue(np.allclose(T_zero[key].as_matrix(), T_zero_reconstructed[key].as_matrix()))

    def test_ik_api(self, nR: int = 32, nG: int = 16, samples: int = 32, dof: int = 6):
        """
        Test the inverse kinematics API, i.e., an inverse kinematics functionality that is framework-agnostic and does
        not require the user to know the details of the generative_graphik approach.
        """
        tic = time()
        t_eval = 0
        graphs = [random_revolute_robot_graph(dof) for _ in range(nR)]
        goals = dict.fromkeys(range(nR), None)
        for i, j in itertools.product(range(nR), range(nG)):
            if j == 0:
                goals[i] = []
            q = torch.rand(dof + 1) * 2 * torch.pi - torch.pi
            angles = {jnt: q_jnt.item() for jnt, q_jnt in zip(graphs[i].robot.joint_ids, q)}
            T = graphs[i].robot.pose(angles, graphs[i].robot.end_effectors[-1])
            goals[i].append(torch.Tensor(T.as_matrix()).to(self.device))

        all_cost = list()
        best_cost = list()
        for i, g in enumerate(graphs):
            T_zero_native = g.robot.from_dh_params(g.robot.params)
            transforms = joint_transforms_from_t_zeros(T_zero_native, keys=g.robot.joint_ids, device=self.device)
            transforms = torch.unsqueeze(transforms, 0)
            sol = ik(transforms, torch.stack(goals[i]), samples=samples, return_all=True)

            t_eval_start = time()

            trans_errors = list()
            rot_errors = list()

            for k, l in itertools.product(range(nG), range(samples)):
                q_kl = {jnt: sol[0, k, l, m].item() for m, jnt in enumerate(g.robot.joint_ids[1:])}
                q_kl['p0'] = 0
                T = g.robot.pose(q_kl, g.robot.end_effectors[-1]).as_matrix()
                e_rot, e_trans = self.ik_error(goals[i][k], torch.tensor(T, device=self.device, dtype=torch.float32))
                rot_errors.append(e_rot)
                trans_errors.append(e_trans)

            cost = [trans + rot * 0.05 / 2 for trans, rot in zip(trans_errors, rot_errors)]  # 2° ~ 0.05m
            # Get at least one good solution
            self.assertLessEqual(np.min(rot_errors), 1)
            self.assertLessEqual(np.min(trans_errors), 0.03)
            self.assertLessEqual(np.min(cost), 0.05)

            # Is it significantly better than random? (educated guess of what a random precision would be)
            self.assertLessEqual(np.mean(rot_errors), 45)
            self.assertLessEqual(np.min(trans_errors),
                                 np.mean([np.linalg.norm(goals[i][j][:3, 3].detach().cpu().numpy()) for j in range(nG)]) / 10)
            best_cost.append(np.min(cost))
            all_cost.append(np.mean(cost))
            t_eval += time() - t_eval_start

        toc = time()
        delta_t = toc - tic - t_eval
        print(f"\n\nIK Test took {delta_t:.2f} seconds. That's {1000 * delta_t / (nR * nG):.2f} ms per goal "
              f"or {1000 * delta_t / nR:.2f} ms per robot.")
        print(f"Mean best: {np.mean(best_cost):.4f}, Std: {np.std(best_cost):.4f}")
        print(f"Mean cost: {np.mean(all_cost):.3f}, Std: {np.std(all_cost):.3f}")


if __name__ == '__main__':
    unittest.main()
