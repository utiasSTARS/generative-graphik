from typing import Optional
import random
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions import Normal, Categorical
from scipy.sparse import find
from liegroups.torch import SE3, SE2, SO3, SO2

from graphik.utils import (
    adjacency_matrix_from_graph, 
    distance_matrix_from_graph,
    TYPE,
    POS
)

TYPE_ENUM = {
    "base": np.asarray([1, 0, 0]),
    "robot": np.asarray([0, 1, 0]),
    "obstacle": np.asarray([0, 0, 1]),
}

def repeat_offset_index(index, repeat, offset):
    """Repeat and offset indices"""
    cumsum = 0
    new_edge_index = []
    for _ in range(repeat):
        new_edge_index.append((index + cumsum).unsqueeze(1))
        cumsum += offset
    new_edge_index = torch.cat(new_edge_index, dim=1)
    return new_edge_index

def get_norm_layer(out_channels, num_groups=32, layer_type='None', layer_dim='1d'):
    if layer_type == 'BatchNorm':
        if layer_dim == '2d':
            return nn.BatchNorm2d(out_channels)
        elif layer_dim == '1d':
            return nn.BatchNorm1d(out_channels)
    elif layer_type == 'GroupNorm':
        return nn.GroupNorm(num_groups, out_channels)
    elif layer_type == 'LayerNorm':
        if layer_dim == '2d':
            return nn.GroupNorm(1, out_channels)
        elif layer_dim == '1d':
            return nn.LayerNorm(out_channels)
    elif layer_type == 'None':
        return nn.Identity()
    else:
        raise NotImplementedError(f"{layer_type} is not a valid normalization layer.")
    

def kl_divergence(d1, d2, K=128):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def MixtureGaussianDiag(categorical_prob, loc, scale, batch_dim=1):
    return MixtureSameFamily(
        Categorical(categorical_prob), MultivariateNormalDiag(loc,scale), batch_dim
    )


def MultivariateNormalDiag(loc, scale, batch_dim=1):
    """Returns a diagonal multivariate normal Torch distribution."""
    return Independent(Normal(loc, scale), batch_dim)


def set_seed_torch(seed):
    """Set the same random seed for all sources of rng?"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def torch_log_from_T(T):
    if T.dim() < 3:
        T = T.unsqueeze(0)

    if T.shape[-1] == 4:
        return SE3(SO3(T[:,:-1,:-1]), T[:,:-1,-1]).log()
    elif T.shape[-1] == 3:
        return SE2(SO2(T[:,:-1,:-1]), T[:,:-1,-1]).log()

def torch_T_from_log(S):
    if S.dim() < 2:
        S = S.unsqueeze(0)

    if S.shape[-1] == 6:
        return SE3.exp(S).as_matrix().reshape(-1, 4, 4)
    elif S.shape[-1] == 3:
        return SE2.exp(S).as_matrix().reshape(-1, 3, 3)

def torch_T_from_log_angle_axis(S,q):
    if S.dim() < 2:
        S = S.unsqueeze(0)
        q = q.unsqueeze(0)

    dim_tangent = S.shape[-1]
    dim = torch.div(dim_tangent, 3, rounding_mode="trunc") + 1
    if dim == 2:
        Sq = torch.mul(q.view(-1, 1), S.view(-1, dim_tangent))
        T = torch_T_from_log(Sq).reshape(-1, dim+1, dim+1)
    else:
        Phi = torch.zeros(S.shape[0], dim, dim, device=S.device)
        Phi[:, 0, 1] = -S[:, 5]
        Phi[:, 1, 0] = S[:, 5]
        Phi[:, 0, 2] = S[:, 4]
        Phi[:, 2, 0] = -S[:, 4]
        Phi[:, 1, 2] = -S[:, 3]
        Phi[:, 2, 1] = S[:, 3]

        Phi2 = torch.bmm(Phi,Phi)

        I = torch.eye(3, device=S.device).unsqueeze(0).expand(S.shape[0],-1,-1)
        T = torch.eye(4, device=S.device).unsqueeze(0).repeat(S.shape[0],1,1)

        sin_q = torch.sin(q)
        cos_q = torch.cos(q)

        A = sin_q * Phi.view(S.shape[0],-1) + (1-cos_q) * Phi2.view(S.shape[0],-1)
        T[:,:dim,:dim] = I + A.view(S.shape[0],3,3)

        B = q * I.view(S.shape[0],-1) + (1 - cos_q) * Phi.view(S.shape[0],-1) + (q - sin_q) * Phi2.view(S.shape[0],-1)
        T[:,:dim,dim] = torch.matmul(B.view(S.shape[0],3,3), S[:,:dim].view(S.shape[0],3,1)).view(S.shape[0],3)

    return T

def edge_indices_attributes(G):
    A = adjacency_matrix_from_graph(G)  # adjacency matrix
    idx, jdx, _ = find(A)  # indices of non-zero elements
    edges = torch.tensor([list(idx), list(jdx)], dtype=torch.long)

    D = np.sqrt(distance_matrix_from_graph(G))  # distance matrix
    d = D[A > 0][np.newaxis,:]
    distances = torch.tensor(d, dtype=torch.float).t()

    return edges, distances

def node_attributes(G, attrs=["pos"]):

    out = []
    for attr in attrs:
        if attr in [POS, "S"]:
            out += [
                torch.tensor(
                    [list(data) for node, data in G.nodes(data=attr)], dtype=torch.float
                )
            ]
        if attr in ["T0"]:
            Ts = np.stack([data.as_matrix() for node, data in G.nodes(data=attr)])
            out += [torch.tensor(Ts, dtype=torch.float)]
        if attr in [TYPE]:
            out += [
                torch.tensor(
                    [list(TYPE_ENUM[data[0]]) for node, data in G.nodes(data=attr)],
                    dtype=torch.float,
                )
            ]
    return tuple(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def SE3_from(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    # Composes SE3 matrix from rotation and translation component
    # ----------------------------------------------------------------------
    T = torch.eye(4, device=rot.device).repeat(rot.shape[0],1,1)
    T[:,:-1,:-1] = rot
    T[:,:-1,-1] = trans
    return T

def SE3_inv_from(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    # Composes inverse SE3 matrix from rotation and translation component
    # ----------------------------------------------------------------------
    inv_rot = rot.transpose(2,1)
    inv_trans = -torch.bmm(inv_rot, trans[:,:,None])[:,:,0]
    inv_T = torch.eye(4, device=rot.device).repeat(rot.shape[0],1,1)
    inv_T[:,:-1,:-1] = inv_rot
    inv_T[:,:-1,-1] = inv_trans
    return inv_T

def SE3_inv(T: torch.Tensor):
    # Computes SE3 matrix inverse
    # ----------------------------------------------------------------------
    return SE3_inv_from(T[:,:-1,:-1], T[:,:-1,-1])

def batchJointScrews(T0: torch.Tensor):
    omega = T0[:,:-1,2] # z axis
    q = T0[:,:-1,-1]
    v = torch.cross(-omega, q)
    return torch.cat([v, omega], dim=-1)


def batchIKmultiDOF(P: torch.Tensor, T0: torch.Tensor, num_joints: torch.Tensor, T_final: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Computes IK for multiple robots with varying DOF from points.
    # ----------------------------------------------------------------------
    # P [sum(num_nodes) x 3] - points corresponding to the distance-geometric
    #   robot model described in (Maric et al.,2021)
    # T0 [num_nodes x 16] - 4x4 matrices of robot frames in home poses
    # num_joints [num_edges x 1] - number of joints (DOF) for each robot in batch
    # ----------------------------------------------------------------------

    device = T0.device # GPU or CPU
    dtype = T0.dtype # data type for non-integer
    dim = P.shape[1] # dimension (2 or 3)

    num_robots = num_joints.shape[0] # total number of robots
    num_nodes = 2*(num_joints+1) + (dim-1) # number of nodes for point graphs
    node_start_ind = torch.cumsum(num_nodes,dim=0) - num_nodes # start indices for nodes
    node_start_ind = node_start_ind.to(device)

    # normalizes the node positions to the canonical coordinate system
    x_hat = (P[node_start_ind + 1] - P[node_start_ind])
    y_hat = -(P[node_start_ind + 2] - P[node_start_ind])
    z_hat = (P[node_start_ind + 3] - P[node_start_ind])

    # get modifies base frames
    R = torch.cat([x_hat.unsqueeze(1), y_hat.unsqueeze(1), z_hat.unsqueeze(1)], dim = 1).transpose(2,1)
    B_inv = SE3_inv_from(R, P[node_start_ind])
    hl = torch.arange(num_robots, device=device).repeat_interleave(num_joints + 1, dim=0)

    mask = torch.tensor(True, device=device).repeat(num_nodes.sum())
    mask[node_start_ind+1] = False
    mask[node_start_ind+2] = False
    P = P.masked_select(mask[:,None].expand(-1,3)).reshape(-1,3)

    num_frames_total = T0.shape[0] # total number of frames on robots
    T = torch.eye(dim+1, device=device, dtype=dtype)[None,:,:].repeat(num_frames_total,1,1)
    theta = torch.zeros(num_frames_total, dtype=dtype, device=device)

    # constant matrices
    T0_inv = SE3_inv(T0) # inverses of T0
    T_rel = torch.bmm(T0_inv, T0.roll(-1,0)) # relative xf between T0
    T0_q = SE3_from(T0[:,:-1,:-1], T0[:,:-1,-1] + T0[:,:-1,-2])
    qs_0 = torch.bmm(T0_inv, T0_q.roll(-1,0))[:,:-1,-1] # qs at home config

    # indices of relevant p and q nodes
    ind = torch.arange(num_frames_total, device=device)
    idx_p = 2*(ind - 1) + 2
    idx_q = 2*(ind - 1) + 3

    # compute normalized q (i.e., distance fixed to 1) and transform to base frame
    q = torch.baddbmm(B_inv[hl[ind],:-1,-1][:,:,None], B_inv[hl[ind],:-1,:-1], P[idx_q][:,:,None])[:,:,0]

    omega_z = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=dtype, device=device)
    omega_z_sq = torch.mm(omega_z, omega_z.transpose(1,0))
    A = torch.matmul(qs_0, omega_z)[:,None,:]
    B = torch.matmul(qs_0, omega_z_sq)[:,None,:]

    # generate virtual edge indices used to multiply pairs of joints
    joint_end_ind = torch.cumsum(num_joints + 1, dim=0) - 1 # end indices of joints
    joint_start_ind =  joint_end_ind - num_joints # start indices of joints

    ei = torch.zeros(2, joint_start_ind.size(-1), dtype=torch.int64, device=device)
    ei[0] = joint_start_ind + 1 # starting jnt id repeat for all con
    ei[1] = joint_end_ind
    inc = torch.tensor([[1],[0]], dtype=torch.int64, device=device) # tensor for inc
    for _ in range(1, num_joints.max()):
        # q point expressed in previous frame
        qs = torch.bmm(T[ei[0]-1,:-1,:-1].transpose(2,1), (q[ei[0]] - T[ei[0]-1,:-1,-1])[:,:,None])[:,:,0]

        # compute angle approximation
        theta[ei[0]-1] = torch.atan2(
            -torch.bmm(A[ei[0]-1], qs[:,None,:].transpose(2,1))[:,0] + 1e-7,
            torch.bmm(B[ei[0]-1], qs[:,None,:].transpose(2,1))[:,0] + 1e-7
        ).reshape(-1)

        rotz = SO3.rotz(theta[ei[0]-1]).as_matrix().reshape(-1,3,3)
        rotmat = SE3_from(rotz, torch.zeros([ei[0].size(0), 3]))
        T[ei[0]] = torch.bmm(torch.bmm(T[ei[0]-1], rotmat), T_rel[ei[0]-1].expand(ei[0].size(-1),-1,-1))

        ei, _ = remove_self_loops(ei + inc)

    ind = torch.arange(num_joints.sum(), device=device) + torch.arange(num_robots, device=device).repeat_interleave(num_joints)

    if T_final is not None:
        # parallel = torch.linalg.cross(T_rel[joint_end_ind-1,:-1,-1], torch.tensor([[0,0,1]], device=device, dtype=dtype)).norm(dim=-1) < 1e-6
        # parallel_idx = parallel.nonzero()
        T_th = torch.bmm(SE3_inv(T[joint_end_ind-1]), T_final)
        theta[joint_end_ind-1] = theta[joint_end_ind-1] +  torch.atan2(T_th[:, 1, 0], T_th[:, 0, 0])

    return theta[ind]

def batchPmultiDOF(T: torch.Tensor, num_joints: torch.Tensor) -> torch.Tensor:
    # from frame transforms to batched points
    # the key problem is to sort everything properly

    device = T.device # GPU or CPU
    dtype = T.dtype # data type for non-integer
    dim = 3 # problem dimension

    num_frames_total = T.shape[0] # total number of frames on robots
    num_nodes = 2*(num_joints+1) + (dim-1) # number of nodes for point graphs
    node_start_ind = torch.cumsum(num_nodes,dim=0) - num_nodes # start indices for nodes

    # indices of relevant p and q n
    ind = torch.arange(num_frames_total, device=device)
    idx_p = 2*(ind - 1) + 2
    idx_q = 2*(ind - 1) + 3

    pos_p = T[:,:-1,-1]
    pos_q = T[:,:-1,-1] + T[:,:-1,-2]
    pos_pq = torch.zeros(2*num_frames_total, 3, dtype=dtype, device=device)
    pos_pq[torch.cat([idx_p, idx_q], dim=-1), :] = torch.cat([pos_p, pos_q], dim=0)


    P = torch.zeros(num_nodes.sum(),3)

    mask = torch.tensor(True, device=device).repeat(num_nodes.sum())
    mask[node_start_ind+1] = False
    mask[node_start_ind+2] = False
    P = torch.masked_scatter(P, mask[:,None].expand(-1,3), pos_pq)
    P[node_start_ind + 1] = torch.tensor([1,0,0], dtype=dtype, device=device)
    P[node_start_ind + 2] = torch.tensor([0,-1,0], dtype=dtype, device=device)

    return P

def batchFKmultiDOF(T0: torch.Tensor, q: torch.Tensor, num_joints: torch.Tensor) -> torch.Tensor:
    # Computes FK for multiple robots with varying DOF using graphs.
    # Uses the lie group FK formula:
    # T_ee = exp(S0*q0)*exp(S1*q1)*...*exp(Sn*qn)*M
    # ----------------------------------------------------------------------
    # T0 [num_nodes x 16] - 4x4 matrices of robot frames in home poses
    # q [num_edges x 1] - joint angles
    # num_joints [num_robots] - number of joints per robot.
    #   This is the the joint-based model description so it's num_joints+1
    # edge_index - all edges in batch
    # ----------------------------------------------------------------------

    device = T0.device
    num_frames = num_joints + 1
    num_robots = num_frames.shape[0]
    total_num_frames = T0.shape[0]
    max_num_frames = num_frames.max()
    frame_start_ind = num_frames.cumsum(dim=0) - num_frames

    # get edge index
    ind_i = torch.arange(num_joints.sum(), device=device) + torch.arange(num_robots, device=device).repeat_interleave(num_joints)
    ind_j = ind_i + 1
    edge_index = torch.concat([ind_i, ind_j], dim=0).reshape(2,-1)
    total_num_edges = edge_index.shape[-1]

    # compute joint screws
    S = batchJointScrews(T0[edge_index[0]])
    dim_tangent = S.shape[-1]
    dim = (dim_tangent // 3) + 1

    # get Tq = exp(S0*q) for every joint for every robot, shape [total_num_nodes x 4 x 4]
    Tq = torch.empty(total_num_frames, dim+1, dim+1, device=device)
    Tq[edge_index[0]] = torch_T_from_log_angle_axis(S, q.view(total_num_edges, 1))

    T0 = T0.reshape(total_num_frames, dim+1, dim+1)
    T = torch.eye(dim+1, device=device).unsqueeze(0).repeat(total_num_frames,1,1)

    ei = torch.zeros(edge_index.shape[0], total_num_frames, dtype=torch.int64, device=device)
    ei[0] = frame_start_ind.repeat_interleave(num_frames) # starting jnt id repeat for all con
    ei[1] = torch.arange(0, total_num_frames) # all others
    inc = torch.tensor([[1],[0]], dtype=torch.int64, device=device) # tensor for inc
    for _ in range(max_num_frames-1):
            ei, _ = remove_self_loops(ei) # remove self-loops for starting
            T[ei[1]] = torch.bmm(T[ei[1]], Tq[ei[0]]) # apply action from prev to cur
            ei = ei + inc # next
    T[edge_index[1]] = torch.bmm(T[edge_index[1]], T0[edge_index[1]])

    return T  