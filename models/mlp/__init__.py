from typing import Type
from .mlp import MLPOutput, MLPBase, MLP, GlobalMLP
from .time_attn_mlp import TimeAttnMLP
from .transformer import AttnMLP
from .pointnet import PointNet, PointNetLG


mlp_dict: dict[str, Type[MLPBase]] = {
    'mlp': MLP,
    'globalmlp': GlobalMLP,
    'timeattnmlp': TimeAttnMLP,
    'attnmlp': AttnMLP,
    'pointnet': PointNet,
    'pointnetlg': PointNetLG,
}

del Type
