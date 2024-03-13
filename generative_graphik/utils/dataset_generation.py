from typing import Iterable, List, Union

from liegroups.numpy.se2 import SE2Matrix
from liegroups.numpy.se3 import SE3Matrix
import numpy as np
import os
from tqdm import tqdm
from dataclasses import dataclass, fields

import torch
from torch_geometric.data import InMemoryDataset, Data
import torch.multiprocessing as mp

import graphik
from graphik.robots import RobotRevolute
from graphik.graphs import ProblemGraphRevolute
from graphik.graphs.graph_revolute import random_revolute_robot_graph
import generative_graphik
from generative_graphik.args.parser import parse_data_generation_args
from generative_graphik.utils.torch_utils import (
    batchFKmultiDOF,
    batchPmultiDOF,
    edge_indices_attributes,
    node_attributes,
)
from graphik.utils import (
    BASE,
    DIST,
    ROBOT,
    OBSTACLE,
    POS,
    TYPE,
    distance_matrix_from_graph
)
from graphik.utils.roboturdf import RobotURDF
import networkx as nx

TYPE_ENUM = {
    BASE: np.asarray([1, 0, 0]),
    ROBOT: np.asarray([0, 1, 0]),
    OBSTACLE: np.asarray([0, 0, 1]),
}
ANCHOR_ENUM = {"anchor": np.asarray([1, 0]), "not_anchor": np.asarray([0, 1])}


class CachedDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super(CachedDataset, self).__init__(None)
        self.data, self.slices = data, slices


@dataclass
class StructData:
    type: Union[List[torch.Tensor], torch.Tensor]
    num_joints: Union[List[int], int]
    num_nodes: Union[List[int], int]
    num_edges: Union[List[int], int]
    partial_mask: Union[List[torch.Tensor], torch.Tensor]
    partial_goal_mask: Union[List[torch.Tensor], torch.Tensor]
    edge_index_full: Union[List[torch.Tensor], torch.Tensor]
    T0: Union[List[torch.Tensor], torch.Tensor]


def create_dataset_from_data_points(data_points: Iterable[Data]) -> CachedDataset:
    """Takes an iterable of Data objects and returns a CachedDataset by concatenating them."""
    data = tuple(data_points)
    types = torch.cat([d.type for d in data], dim=0)
    T0 = torch.cat([d.T0 for d in data], dim=0).reshape(-1, 4, 4)
    device = T0.device
    num_joints = torch.concat([d.num_joints for d in data])
    num_nodes = torch.tensor([d.num_nodes for d in data], device=device)
    num_edges = torch.tensor([d.num_edges for d in data], device=device)

    P = torch.cat([d.pos for d in data], dim=0)
    distances = torch.cat([d.edge_attr for d in data], dim=0)
    T_ee = torch.stack([d.T_ee for d in data], dim=0)
    masks = torch.cat([d.partial_mask for d in data], dim=-1)
    edge_index_full = torch.cat([d.edge_index_full for d in data], dim=-1)
    partial_goal_mask = torch.cat([d.partial_goal_mask for d in data], dim=-1)

    node_slice = torch.cat([torch.tensor([0], device=device), (num_nodes).cumsum(dim=-1)])
    joint_slice = torch.cat([torch.tensor([0], device=device), (num_joints).cumsum(dim=-1)])
    frame_slice = torch.cat([torch.tensor([0], device=device), (num_joints + 1).cumsum(dim=-1)])
    robot_slice = torch.arange(num_joints.size(0) + 1, device=device)
    edge_full_slice = torch.cat([torch.tensor([0], device=device), (num_edges).cumsum(dim=-1)])

    slices = {
        "edge_attr": edge_full_slice,
        "pos": node_slice,
        "type": node_slice,
        "T_ee": robot_slice,
        "num_joints": robot_slice,
        "partial_mask": edge_full_slice,
        "partial_goal_mask": node_slice,
        "edge_index_full": edge_full_slice,
        "M": frame_slice,
        "q_goal": joint_slice,
    }

    data = Data(
        type=types,
        pos=P,
        edge_attr=distances,
        T_ee=T_ee,
        num_joints=num_joints.type(torch.int32),
        partial_mask=masks,
        partial_goal_mask=partial_goal_mask,
        edge_index_full=edge_index_full.type(torch.int32),
        M=T0,
    )

    return CachedDataset(data, slices)

def generate_data_point_from_pose(graph, T_ee, device = None) -> Data:
    """
    Generates a data point (~problem) from a problem graph and a desired end-effector pose.
    """
    if isinstance(T_ee, torch.Tensor):
        if device is None:
            device = T_ee.device
        T_ee = T_ee.detach().cpu().numpy()
    if isinstance(T_ee, np.ndarray):
        if T_ee.shape == (4, 4):
            T_ee = SE3Matrix.from_matrix(T_ee, normalize=True)
        else:
            raise ValueError(f"Expected T_ee to be of shape (4, 4) or be SEMatrix, got {T_ee.shape}")
    struct_data = generate_struct_data(graph, device)
    num_joints = torch.tensor([struct_data.num_joints])
    edge_index_full = struct_data.edge_index_full.to(dtype=torch.int32, device=device)
    T0 = struct_data.T0

    # Build partial graph nodes
    G_partial = graph.from_pose(T_ee)
    T_ee = torch.from_numpy(T_ee.as_matrix()).to(dtype=torch.float32, device=device)
    P = np.array([p[1] for p in list(G_partial.nodes.data('pos', default=np.array([0,0,0])))])
    P = torch.from_numpy(P).to(dtype=torch.float32, device=device)

    # Build distances of partial graph
    distances = np.sqrt(distance_matrix_from_graph(G_partial))
    # Remove self-loop
    distances = distances[~np.eye(distances.shape[0],dtype=bool)].reshape(distances.shape[0],-1)
    distances = torch.from_numpy(distances).to(dtype=torch.float32, device=device)
    # Remove filler NetworkX extra 1s
    distances = struct_data.partial_mask * distances.reshape(-1)
    return Data(
        pos=P,
        edge_index_full=edge_index_full,
        edge_attr=distances.unsqueeze(1),
        T_ee=T_ee,
        num_joints=num_joints.to(dtype=torch.int32, device=device),
        q_goal=None,
        partial_mask=struct_data.partial_mask,
        partial_goal_mask=struct_data.partial_goal_mask,
        type=struct_data.type,
        T0=struct_data.T0,
    )

def generate_data_point(graph):
    struct_data = generate_struct_data(graph)

    num_joints = torch.tensor([struct_data.num_joints])
    edge_index_full = struct_data.edge_index_full
    T0 = struct_data.T0

    q = torch.rand(num_joints[0], dtype=T0.dtype) * 2 * torch.pi - torch.pi
    q[num_joints[0] - 1] = 0
    T = batchFKmultiDOF(T0, q, num_joints)
    P = batchPmultiDOF(T, num_joints)
    T_ee = T[num_joints[0]]
    distances = torch.linalg.norm(
        P[edge_index_full[0], :] - P[edge_index_full[1], :], dim=-1
    )
    return Data(
        pos=P,
        edge_index_full=edge_index_full.type(torch.int32),
        edge_attr=distances.unsqueeze(1),
        T_ee=T_ee,
        num_joints=num_joints.type(torch.int32),
        q_goal=q,
        partial_mask=struct_data.partial_mask,
        partial_goal_mask=struct_data.partial_goal_mask,
        type=struct_data.type,
        T0=struct_data.T0,
    )


def generate_struct_data(graph, device=None):

    robot = graph.robot
    dof = robot.n
    num_joints = dof
    num_nodes = 2 * (dof + 1) + 2  # number of nodes for point graphs

    type = node_attributes(graph, attrs=[TYPE])[0]
    T0 = node_attributes(graph.robot, attrs=["T0"])[0]

    G_partial = graph.from_pose(robot.pose(robot.random_configuration(), f"p{dof}"))
    edge_index_partial, _ = edge_indices_attributes(G_partial)
    # D = nx.to_scipy_sparse_array(G_partial.to_undirected(), weight=DIST, format="coo")
    # ind0, ind1 = D.row, D.col
    ind0 = edge_index_partial[0]
    ind1 = edge_index_partial[1]

    edge_index_full = (
        (torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes))
        .nonzero()
        .transpose(0, 1)
    )
    num_edges = edge_index_full[-1].shape[-1]

    partial_goal_mask = torch.zeros(num_nodes)
    partial_goal_mask[: graph.dim + 1] = 1
    partial_goal_mask[-2:] = 1

    # _______extracting partial indices from vectorized full indices via mask
    mask_gen = torch.zeros(num_nodes, num_nodes)  # square matrix of zeroes
    mask_gen[ind0, ind1] = 1  # set partial elements to 1
    mask = (
        mask_gen[edge_index_full[0], edge_index_full[1]] > 0
    )  # get full elements from matrix (same order as generated)

    data = StructData(
        type=type,
        num_joints=num_joints,
        num_edges=num_edges,
        num_nodes=num_nodes,
        partial_mask=mask,
        partial_goal_mask=partial_goal_mask,
        edge_index_full=edge_index_full,
        T0=T0,
    )
    if device is None:
        return data
    data = StructData(**{
        f.name: getattr(data, f.name).to(device)
        if isinstance(getattr(data, f.name), torch.Tensor)
        else getattr(data, f.name)
        for f in fields(data)
    })
    return data


def generate_specific_robot_data(robots, num_examples, params):

    examples_per_robot = num_examples // len(robots)

    all_struct_data = StructData(
        type=[],
        num_joints=[],
        num_nodes=[],
        num_edges=[],
        partial_mask=[],
        partial_goal_mask=[],
        edge_index_full=[],
        T0=[],
    )

    q_lim_l_all = []
    q_lim_u_all = []

    for robot_name in robots:
        # generate data for robot like ur10, kuka etc.
        if robot_name == "ur10":
            # randomize won't work on ur10
            # robot, graph = load_ur10(limits=None, randomized_links = False)
            fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
            q_lim_l = -np.pi * np.ones(6)
            q_lim_u = np.pi * np.ones(6)
        elif robot_name == "kuka":
            # robot, graph = load_kuka(limits=None, randomized_links = params["randomize"], randomize_percentage=0.2)
            fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
            q_lim_l = -np.pi * np.ones(7)
            q_lim_u = np.pi * np.ones(7)
        elif robot_name == "lwa4d":
            # robot, graph = load_schunk_lwa4d(limits=None, randomized_links = params["randomize"], randomize_percentage=0.2)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
            q_lim_l = -np.pi * np.ones(7)
            q_lim_u = np.pi * np.ones(7)
        elif robot_name == "panda":
            # robot, graph = load_schunk_lwa4d(limits=None, randomized_links = params["randomize"], randomize_percentage=0.2)
            fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
            # q_lim_l = -np.pi * np.ones(7)
            # q_lim_u = np.pi * np.ones(7)
            q_lim_l = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            q_lim_u = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        elif robot_name == "lwa4p":
            # robot, graph = load_schunk_lwa4p(limits=None, randomized_links = params["randomize"], randomize_percentage=0.2)
            fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
            q_lim_l = -np.pi * np.ones(6)
            q_lim_u = np.pi * np.ones(6)
        else:
            raise NotImplementedError

        urdf_robot = RobotURDF(fname)
        robot = urdf_robot.make_Revolute3d(
            q_lim_l,
            q_lim_u,
            randomized_links=params["randomize"],
            randomize_percentage=params["randomize_percentage"],
        )  # make the Revolute class from a URDF
        graph = ProblemGraphRevolute(robot)
        struct_data = generate_struct_data(graph)

        for _ in tqdm(range(examples_per_robot), leave=False):
            # q_lim_l_all.append(q_lim_l)
            # q_lim_u_all.append(q_lim_u)
            for field in struct_data.__dataclass_fields__:
                all_struct_data.__dict__[field].append(getattr(struct_data, field))

    types = torch.cat(all_struct_data.type, dim=0)
    T0 = torch.cat(all_struct_data.T0, dim=0).reshape(-1, 4, 4)
    # q_lim_l_all = torch.from_numpy(np.concatenate(q_lim_l_all)).type(T0.dtype)
    # q_lim_u_all = torch.from_numpy(np.concatenate(q_lim_u_all)).type(T0.dtype)
    num_joints = torch.tensor(all_struct_data.num_joints)
    num_nodes = torch.tensor(all_struct_data.num_nodes)
    num_edges = torch.tensor(all_struct_data.num_edges)

    # problem is that edge_index_full doesn't contain self-loops
    masks = torch.cat(all_struct_data.partial_mask, dim=-1)
    edge_index_full = torch.cat(all_struct_data.edge_index_full, dim=-1)
    partial_goal_mask = torch.cat(all_struct_data.partial_goal_mask, dim=-1)

    # delete struct data
    all_struct_data = None
    q = torch.rand(num_joints.sum(), dtype=T0.dtype) * 2 * torch.pi - torch.pi
    # q = torch.rand(num_joints.sum(), dtype=T0.dtype) * (q_lim_u_all - q_lim_l_all) + q_lim_l_all
    
    q[(num_joints).cumsum(dim=-1) - 1] = 0
    T = batchFKmultiDOF(T0, q, num_joints)
    P = batchPmultiDOF(T, num_joints)
    # T_ee = T[num_joints.cumsum(dim=-1)]
    T_ee = T[torch.cumsum(num_joints + 1, dim=0) - 1]
    offset_full = (
        torch.cat([torch.tensor([0]), num_nodes[:-1].cumsum(dim=-1)])
        .repeat_interleave(num_edges, dim=-1)
        .unsqueeze(0)
        .expand(2, -1)
    )
    edge_index_full_offset = edge_index_full + offset_full
    distances = torch.linalg.norm(
        P[edge_index_full_offset[0], :] - P[edge_index_full_offset[1], :], dim=-1
    )

    node_slice = torch.cat([torch.tensor([0]), (num_nodes).cumsum(dim=-1)])
    joint_slice = torch.cat([torch.tensor([0]), (num_joints).cumsum(dim=-1)])
    frame_slice = torch.cat([torch.tensor([0]), (num_joints + 1).cumsum(dim=-1)])
    robot_slice = torch.arange(num_joints.size(0) + 1)
    edge_full_slice = torch.cat([torch.tensor([0]), (num_edges).cumsum(dim=-1)])

    slices = {
        "edge_attr": edge_full_slice,
        "pos": node_slice,
        "type": node_slice,
        "T_ee": robot_slice,
        "num_joints": robot_slice,
        "partial_mask": edge_full_slice,
        "partial_goal_mask": node_slice,
        "edge_index_full": edge_full_slice,
        "M": frame_slice,
        "q_goal": joint_slice,
    }

    data = Data(
        type=types,
        pos=P,
        edge_attr=distances.unsqueeze(1),
        T_ee=T_ee,
        num_joints=num_joints.type(torch.int32),
        partial_mask=masks,
        partial_goal_mask=partial_goal_mask,
        edge_index_full=edge_index_full.type(torch.int32),
        M=T0,
        q_goal=q,
    )
    return data, slices


def generate_random_struct_data(dof):
    return generate_struct_data(random_revolute_robot_graph(dof))


def generate_randomized_robot_data(robot_type, dofs, num_examples, params):
    # generate data for randomized robots

    examples_per_dof = num_examples // len(dofs)
    print("Generating " + robot_type + " data!")

    all_struct_data = StructData(
        type=[],
        num_joints=[],
        num_nodes=[],
        num_edges=[],
        partial_mask=[],
        partial_goal_mask=[],
        edge_index_full=[],
        T0=[],
    )

    for dof in dofs:
        with mp.Pool() as p:
            graphs = p.map(random_revolute_robot_graph, [dof] * examples_per_dof)
        for idx in tqdm(range(examples_per_dof), leave=False):
            struct_data = generate_struct_data(graphs[idx])
            for field in struct_data.__dataclass_fields__:
                all_struct_data.__dict__[field].append(getattr(struct_data, field))

    types = torch.cat(all_struct_data.type, dim=0)
    T0 = torch.cat(all_struct_data.T0, dim=0).reshape(-1, 4, 4)
    num_joints = torch.tensor(all_struct_data.num_joints)
    num_nodes = torch.tensor(all_struct_data.num_nodes)
    num_edges = torch.tensor(all_struct_data.num_edges)

    # problem is that edge_index_full doesn't contain self-loops
    masks = torch.cat(all_struct_data.partial_mask, dim=-1)
    edge_index_full = torch.cat(all_struct_data.edge_index_full, dim=-1)
    partial_goal_mask = torch.cat(all_struct_data.partial_goal_mask, dim=-1)

    # delete struct data
    all_struct_data = None

    q = torch.rand(num_joints.sum(), dtype=T0.dtype) * 2 * torch.pi - torch.pi
    q[(num_joints).cumsum(dim=-1) - 1] = 0
    T = batchFKmultiDOF(T0, q, num_joints)
    P = batchPmultiDOF(T, num_joints)
    T_ee = T[num_joints.cumsum(dim=-1)]
    offset_full = (
        torch.cat([torch.tensor([0]), num_nodes[:-1].cumsum(dim=-1)])
        .repeat_interleave(num_edges, dim=-1)
        .unsqueeze(0)
        .expand(2, -1)
    )
    edge_index_full_offset = edge_index_full + offset_full
    distances = torch.linalg.norm(
        P[edge_index_full_offset[0], :] - P[edge_index_full_offset[1], :], dim=-1
    )

    node_slice = torch.cat([torch.tensor([0]), (num_nodes).cumsum(dim=-1)])
    joint_slice = torch.cat([torch.tensor([0]), (num_joints).cumsum(dim=-1)])
    frame_slice = torch.cat([torch.tensor([0]), (num_joints + 1).cumsum(dim=-1)])
    robot_slice = torch.arange(num_joints.size(0) + 1)
    edge_full_slice = torch.cat([torch.tensor([0]), (num_edges).cumsum(dim=-1)])

    slices = {
        "edge_attr": edge_full_slice,
        "pos": node_slice,
        "type": node_slice,
        "T_ee": robot_slice,
        "num_joints": robot_slice,
        "partial_mask": edge_full_slice,
        "partial_goal_mask": node_slice,
        "edge_index_full": edge_full_slice,
        "M": frame_slice,
        "q_goal": joint_slice,
    }

    data = Data(
        type=types,
        pos=P,
        edge_attr=distances.unsqueeze(1),
        T_ee=T_ee,
        num_joints=num_joints.type(torch.int32),
        partial_mask=masks,
        partial_goal_mask=partial_goal_mask,
        edge_index_full=edge_index_full.type(torch.int32),
        M=T0,
        q_goal=q,
    )

    return data, slices


def generate_dataset(params, robots):
    dof = params.get("dof", [3])  # if no dofs are defined, default to 3
    num_examples = params.get("size", 1000)

    if robots[0] == "revolute_chain":
        data, slices = generate_randomized_robot_data(
            robots[0], dof, num_examples, params
        )
    else:
        data, slices = generate_specific_robot_data(robots, num_examples, params)

    return data, slices


def main(args):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if args.num_examples > args.max_examples_per_file:
        num_files = int(args.num_examples / args.max_examples_per_file)
    else:
        num_files = 1

    if args.storage_base_path is None:
        storage_path = generative_graphik.__path__[0] + "/../datasets/" + args.id + "/"
        val_path = (
            generative_graphik.__path__[0] + "/../datasets/" + args.id + "_validation/"
        )
    else:
        storage_path = args.storage_base_path
        val_path = args.storage_base_path + "_validation/"

    if not os.path.exists(storage_path):
        print(f"Path {storage_path} not found. Creating directory.")
        os.makedirs(storage_path)

    if not os.path.exists(val_path):
        print(f"Path {val_path} not found. Creating directory.")
        os.makedirs(val_path)

    print(f"Saving dataset to {storage_path} as {num_files} separate files.")
    for idx in range(num_files):

        dataset_params = {
            "size": args.num_examples // num_files,
            "samples": args.num_samples,
            "dof": args.dofs,
            "goal_type": args.goal_type,
            "randomize": args.randomize,
            "randomize_percentage": args.randomize_percentage,
        }

        data, slices = generate_dataset(
            dataset_params,
            args.robots,
        )

        dataset = CachedDataset(data, slices)

        with open(os.path.join(storage_path, "data_" + f"{idx}" + ".p"), "wb") as f:
            torch.save(dataset, f)

    num_val_examples = int(
        (args.num_examples / num_files) / (100 / args.validation_percentage)
    )
    print(
        f"Generating validation set with {num_val_examples} problems (10% of single file)."
    )
    dataset_params = {
        "size": num_val_examples,
        "samples": args.num_samples,
        "dof": args.dofs,
        "goal_type": args.goal_type,
        "randomize": args.randomize,
        "randomize_percentage": args.randomize_percentage,
    }
    data, slices = generate_dataset(
        dataset_params,
        args.robots,
    )
    dataset = CachedDataset(data, slices)
    with open(val_path + "data_0" + ".p", "wb") as f:
        torch.save(dataset, f)


if __name__ == "__main__":
    # args = parse_data_generation_args()
    # main(args)
    dataset_params = {
        "size": 10,
        "samples": 100,
        "dof": [6],
        "goal_type": "pose",
        "randomize": False,
        "randomize_percentage": 0.5,
    }
    data, slices = generate_dataset(
        dataset_params,
        ["revolute_chain"],
    )
