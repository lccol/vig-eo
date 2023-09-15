import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from torch import Tensor
from torch_scatter import scatter
from typing import Optional, Union, List, Tuple, Dict

from .utils import graph_to_image, image_to_graph, compute_edge_index, batched_graph_to_graph

class Grapher(gnn.MessagePassing):
    def __init__(self, in_features: int, heads: int, out_features: int, flow: str='target_to_source', **kwargs) -> None:
        super(Grapher, self).__init__(flow=flow, **kwargs)
        self.in_features = in_features
        self.heads = heads
        self.out_features = out_features

        self.w_update = nn.Linear(2 * in_features, heads * out_features) # 2 * in_features because the output is concatenated
        self.reset()
        return

    def reset(self) -> None:
        self.w_update.reset_parameters()
        return

    def forward(self, x: Tensor, edge_index: Tensor, y: Optional[Tensor]=None, size=None) -> Tensor:
        if y is None:
            messages = self.propagate(edge_index, x=(x, x), size=size)
        else:
            messages = self.propagate(edge_index, x=(x, y), size=size)
        res = torch.cat([x, messages], dim=-1)
        res = self.w_update(res)
        return res

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return x_j - x_i

    def aggregate(self, inputs: Tensor, index: Tensor) -> Tensor:
        y = scatter(inputs, index, dim=0, reduce='max')
        return y

class GrapherFC(Grapher):
    def __init__(self, in_features: int, heads: int, out_features: int, reconstruct_image: bool=False, k: int=9, act: str='relu') -> None:
        super(GrapherFC, self).__init__(in_features, heads, out_features)
        self.reconstruct_image = reconstruct_image
        self.k = k
        self.act = act

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        
        if act == 'relu':
            self.act_l = nn.ReLU()
        elif act == 'gelu':
            self.act_l = nn.GeLU()
        elif act == 'leakyrelu':
            self.act_l = nn.LeakyReLU()
        else:
            raise ValueError(f'Activation layer not yet implemented {act}')

        self.fc2 = nn.Sequential(
            nn.Linear(heads * out_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        # TODO implement relative positional encoding

        return

    def forward(self, x: Tensor, y: Optional[Tensor]=None, size=None) -> Tensor:
        B, _, H, W = x.shape
        x = image_to_graph(x, batched=True)
        # x.shape == B, N, C, being N == H * W (1 patch = 1 node)
        if y is None:
            edge_index = compute_edge_index(x, x, k=self.k, is_batched=True)
        else:
            y = image_to_graph(y, batched=True)
            edge_index = compute_edge_index(x, y, self.k, is_batched=True)

        # convert from batched graph to graph
        x = batched_graph_to_graph(x)
        y = batched_graph_to_graph(y)

        x_proj = self.fc1(x)
        x_proj = super(GrapherFC, self).forward(x_proj, edge_index, y, size)
        x_proj = self.act_l(x_proj)
        x = self.fc2(x_proj) + x
        if self.reconstruct_image:
            x = graph_to_image(x, B, H, W)
        return x