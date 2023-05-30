import torch
import torch.nn as nn

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing

from generative_graphik.utils.torch_utils import get_norm_layer

class ResWrapper(nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, :self.dim_res]
        out = self.module(x)
        return out + res

class EGNNLayer(MessagePassing):
    def __init__(
        self,
        non_linearity,
        channels_h,
        channels_m, 
        channels_a,
        aggr: str = 'add', 
        norm_layer: str = 'None',
        hidden_channels: int = 64,
        mlp_layers=2,
        **kwargs
    ):
        super(EGNNLayer, self).__init__(aggr=aggr, **kwargs)
        self.m_len = channels_m

        phi_e_layers = []
        phi_e_layers.extend([
            nn.Linear(2 * channels_h + 1 + channels_a, hidden_channels),
            get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ])
        for _ in range(mlp_layers-2):
            phi_e_layers.extend([
                nn.Linear(hidden_channels, hidden_channels),
                get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ])
        phi_e_layers.extend([
            nn.Linear(hidden_channels, channels_m),
            get_norm_layer(channels_m, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ])
        self.phi_e = nn.Sequential(*phi_e_layers)

        phi_x_layers = []
        phi_x_layers.extend([
            nn.Linear(channels_m, hidden_channels),
            get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ])
        for _ in range(mlp_layers-2):
            phi_x_layers.extend([
                nn.Linear(hidden_channels, hidden_channels),
                get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ])
        phi_x_layers.append(nn.Linear(hidden_channels, 1))
        self.phi_x = nn.Sequential(*phi_x_layers)

        phi_h_layers = []
        phi_h_layers.extend([
            nn.Linear(channels_h + channels_m, hidden_channels),
            get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ])
        for _ in range(mlp_layers-2):
            phi_h_layers.extend([
                nn.Linear(hidden_channels, hidden_channels),
                get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ])
        phi_h_layers.append(nn.Linear(hidden_channels, channels_h))
        self.phi_h = nn.Sequential(*phi_h_layers)
        self.phi_h = ResWrapper(self.phi_h, dim_res=channels_h)

    def forward(self, x, h, edge_attr, edge_index, c=None):
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)

    def message(self, x_i, x_j, h_i, h_j, edge_attr):
        mh_ij = self.phi_e(torch.cat([h_i, h_j, torch.norm(x_i - x_j, dim=-1, keepdim=True)**2, edge_attr], dim=-1))
        mx_ij = (x_i - x_j) * self.phi_x(mh_ij)
        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out, x, h, edge_attr, c):
        m_x, m_h = aggr_out[:, :self.m_len], aggr_out[:, self.m_len:]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1
