import torch
import torch.nn as nn

from generative_graphik.utils.torch_utils import get_norm_layer
from generative_graphik.networks.egnn import EGNNLayer

class EqGraph(nn.Module):
    def __init__(
            self,
            latent_dim=32,
            out_channels_node=3,
            coordinates_dim=3,
            node_features_dim=5,
            edge_features_dim=1,
            mlp_hidden_size=64,
            num_graph_mlp_layers=0,
            num_egnn_mlp_layers=2,
            num_gnn_layers=3,
            stochastic=False,
            num_mixture_components=8,
            norm_layer='None',
            non_linearity=nn.SiLU(),
    ) -> None:
        super(EqGraph, self).__init__()
        self.stochastic = stochastic
        self.num_mixture_components = num_mixture_components
        self.num_gnn_layers = num_gnn_layers
        self.dim_out = out_channels_node

        mlp_x_list = []
        mlp_x_list.extend([
            nn.Linear(coordinates_dim, mlp_hidden_size),
            get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ])
        for _ in range(num_graph_mlp_layers):
            mlp_x_list.extend([
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ])
        mlp_x_list.append(nn.Linear(mlp_hidden_size, latent_dim))
        self.mlp_x = nn.Sequential(*mlp_x_list)

        mlp_h_list = []
        mlp_h_list.extend([
            nn.Linear(node_features_dim, mlp_hidden_size),
            get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ])
        for _ in range(num_graph_mlp_layers):
            mlp_h_list.extend([
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ])
        mlp_h_list.append(nn.Linear(mlp_hidden_size, latent_dim))
        self.mlp_h = nn.Sequential(*mlp_h_list)

        self.gnn = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn.append(
                EGNNLayer(
                    channels_h=latent_dim,
                    channels_m=latent_dim,
                    channels_a=edge_features_dim,
                    hidden_channels=mlp_hidden_size,
                    norm_layer=norm_layer,
                    non_linearity=non_linearity,
                    mlp_layers=num_egnn_mlp_layers
                )
            )

        # Final encoder layer that outputs mean and stddev
        self.fc_mu = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, mlp_hidden_size),
            get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
            non_linearity,
            nn.Linear(mlp_hidden_size, self.num_mixture_components * out_channels_node),
        )

        if self.stochastic:
            self.fc_logvar = nn.Sequential(
                nn.Linear(latent_dim + latent_dim, mlp_hidden_size),
                get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
                non_linearity,
                nn.Linear(mlp_hidden_size, self.num_mixture_components * out_channels_node),
            )

        if self.num_mixture_components > 1:
            self.fc_mixture = nn.Sequential(
                nn.Linear(latent_dim + latent_dim, mlp_hidden_size),
                get_norm_layer(mlp_hidden_size, layer_type=norm_layer, layer_dim="1d"),
                non_linearity,
                nn.Linear(mlp_hidden_size, self.num_mixture_components),
                nn.Softmax(dim=1)
            )

    def forward(self, x, h, edge_attr, edge_index, c=None):
        n = x.shape[0]
        x_l = self.mlp_x(x)
        h_l = self.mlp_h(h)

        for ii in range(self.num_gnn_layers):
            x_l1, h_l1 = self.gnn[ii](
                x=x_l,
                h=h_l,
                edge_attr=edge_attr,
                edge_index=edge_index,
                c=c
            )
            x_l = x_l1
            h_l = h_l1
        v = torch.cat([x_l, h_l], dim=-1)

        mu = self.fc_mu(v)
        mu = mu.reshape(n, self.num_mixture_components, self.dim_out)
        if self.stochastic:
            logvar = self.fc_logvar(v)
            std = torch.exp(logvar / 2.0)
            std = std.reshape(n, self.num_mixture_components, self.dim_out)
            if self.num_mixture_components > 1:
                k = self.fc_mixture(v)
                return (k, mu, std)
            else:
                return (mu.squeeze(1), std.squeeze(1))
        else:
            return mu.squeeze(1)