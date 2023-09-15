import torch
import torch.nn as nn

from torch import Tensor

from typing import Union, List, Dict, Tuple, Optional

def get_activation_layer(act: str, **kwargs):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU(**kwargs)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'softmax':
        return nn.Softmax(**kwargs)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)
    else:
        raise NotImplementedError(f'Activation layer not implemented yet {act}')

def convert_neigh_list_to_edge_index(
        nn_idx: Tensor,
        y_max: Optional[int]=None,
        is_batched: bool=True
    ) -> Tensor:
    # nn_idx is of shape B, N, K is if_batched == True, being k the number of neighbors
    # the output should be of shape 2, |E|
    if is_batched:
        B, N, K = nn_idx.shape
    else:
        N, K = nn_idx.shape

    def _compute_offset(n: int) -> Tensor:
        # n is the maximum index value admitted (= number of columns in the distance matrix)
        offset = (torch.arange(B).to(nn_idx.device) * n).reshape(B, 1, 1)
        offset = offset.repeat(1, N, K)
        return offset
        
    with torch.no_grad():
        if is_batched:
            # compute an index offset to be added to every entry
            src_offset = _compute_offset(N)

            if y_max is None:
                dst_offset = src_offset
            else:
                dst_offset = _compute_offset(y_max)

            src_idx = torch.arange(N).to(nn_idx.device).reshape(1, N, 1)
            src_idx = src_idx.repeat(B, 1, K)

            # add offset to both the central index and the neighbor index
            src_idx += src_offset
            nn_idx += dst_offset

            edge_index = torch.stack([src_idx, nn_idx], dim=1).to(nn_idx.device) # B, 2, N, K
            edge_index = edge_index.flatten(2) # B, 2, |E| per batch
            edge_index = edge_index.permute(1, 2, 0).contiguous() # 2, |E| per batch, B
            edge_index = edge_index.flatten(1).contiguous() # 2, |E|
        else:
            src_idx = torch.arange(N).to(nn_idx.device).unsqueeze(-1).repeat(1, K)
            edge_index = torch.stack([src_idx, nn_idx], dim=0).to(nn_idx.device) # 2, N, K
            edge_index = edge_index.flatten(1).contiguous() # 2, |E|

    return edge_index

def compute_edge_index(x: Tensor, y: Tensor, k: int, distance: str='pointcloud', is_batched: bool=True) -> Tensor:
    assert distance in {'pointcloud', 'euclidean'}, f'Distance metric invalid ({distance})'
    #TODO add relative positional encoding
    with torch.no_grad():
        if distance == 'pointcloud':
            dist = pointcloud_dist(x, y, is_batched)
        elif distance == 'euclidean':
            dist = euclidean_dist(x, y)
        else:
            raise NotImplementedError(f"Distance {distance} not implemented yet")
        _, nn_idx = torch.topk(dist, k=k, largest=False)
        edge_index = convert_neigh_list_to_edge_index(nn_idx, y.shape[-2], is_batched)
    return edge_index

def pointcloud_dist(x: Tensor, y: Tensor, is_batched: bool=True) -> Tensor:
    # This function comes from
    # https://github.com/huawei-noah/Efficient-AI-Backbones/blob/860e89a0fdb45f55510fa5c3a5580a9d3afd69eb/vig_pytorch/gcn_lib/torch_edge.py#L39
    # All credits to the authors of the ViG Paper "Vision GNN: An Image is Worth Graph of Nodes"
    # https://arxiv.org/abs/2206.00272
    features_index, node_index = (2, 1) if is_batched else (1, 0)
    with torch.no_grad():
        xy_inner = torch.matmul(x, y.transpose(features_index, node_index))
        x_square = (x * x).sum(dim=-1, keepdim=True)
        y_square = (y * y).sum(dim=-1, keepdim=True).transpose(features_index, node_index)
        return x_square - 2 * xy_inner + y_square

def euclidean_dist(x: Tensor, y: Optional[Tensor]) -> Tensor:
    with torch.no_grad():
        return torch.cdist(x, y, p=2)

def image_to_graph(x: Tensor, batched=True) -> Tensor:
    # B, C, H, W -> B, N, C -> NB, C
    B, C, H, W = x.shape
    x = x.flatten(2).permute(0, 2, 1)
    if batched:
        # B, H*W, C
        return x.contiguous()
    else:
        return x.reshape(-1, C).contiguous()
    
def batched_graph_to_graph(x: Tensor) -> Tensor:
    # x.shape == B, N, C
    # N = H * W
    # desired shape: |V|, C
    x = x.flatten(0, 1)
    return x

def graph_to_image(x: Tensor, b: int, h: int, w: int) -> Tensor:
    c = x.shape[1]
    # NB, C -> B, N, C -> B, C, N -> B, C, H, W
    x = x.reshape(b, -1, c) \
        .transpose(1, 2) \
        .reshape(b, -1, h, w).contiguous()
    return x