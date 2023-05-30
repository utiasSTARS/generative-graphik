import torch
import torch.nn as nn

from generative_graphik.utils.torch_utils import Flatten, get_norm_layer

class LinearVAE(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_layer='None',
        dropout=False,
        non_linearity=nn.SiLU(),
        hidden_size=256,
        stochastic=False,
        num_mixture_components=1
    ):
        super(LinearVAE, self).__init__()
        self.flatten = Flatten()
        self.stochastic = stochastic
        self.layers = nn.ModuleList()
        self.num_mixture_components = num_mixture_components
        self.dim_out = dim_out

        self.layers.append(nn.Linear(dim_in, hidden_size))
        self.layers.append(get_norm_layer(hidden_size, layer_type=norm_layer, layer_dim="1d"))
        if dropout: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(non_linearity)

        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            get_norm_layer(hidden_size, layer_type=norm_layer, layer_dim="1d"),
            non_linearity,
            nn.Linear(hidden_size, num_mixture_components * dim_out),
        )
        if self.stochastic:
            self.fc_logvar = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                get_norm_layer(hidden_size, layer_type=norm_layer, layer_dim="1d"),
                non_linearity,
                nn.Linear(hidden_size, num_mixture_components * dim_out)
            )

        if self.num_mixture_components > 1:
            self.fc_mixture = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                get_norm_layer(hidden_size, layer_type=norm_layer, layer_dim="1d"),
                non_linearity,
                nn.Linear(hidden_size, num_mixture_components),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        n = x.shape[0]
        for l in self.layers:
            x = l(x)

        mu = self.fc_mu(x)
        mu = mu.reshape(n, self.num_mixture_components, self.dim_out)
        if self.stochastic:
            logvar = self.fc_logvar(x)
            std = torch.exp(logvar / 2.0)
            std = std.reshape(n, self.num_mixture_components, self.dim_out)
            if self.num_mixture_components > 1:
                k = self.fc_mixture(x)
                return (k, mu, std)
            else:
                return (mu.squeeze(1), std.squeeze(1))
        else:
            return mu.squeeze(1)
