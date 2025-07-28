from typing import Literal
import torch
from torch import nn
from .mlp import MLPBase


class PointNet(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        '''
        ***CAUTION: All parameters will be ignored and replaced with the default PointNet configuration.***
        Qi et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
        '''
        nn.Module.__init__(self)
        MLPBase.__init__(self, 4, [64, 64, 64, 128, 1024, 512, 256, 128], 3, 8, nn.ReLU)

        self.encoder = nn.ModuleList([
            self.__make_block(4, 64, 64),
            self.__make_block(64, 64, 128, 1024),
        ])
        self.decoder = self.__make_block(1088, 512, 256, 128)
        self.head = nn.Linear(128, 3)
        
    @property
    def input_shape(self) -> Literal['flat', 'spatio-temporal']:
        return 'spatio-temporal'
        
    def __make_block(
        self,
        in_channels: int, 
        *hid_channels: tuple[int, ...], 
        activation=nn.ReLU, 
        use_bn: bool=True,
    ) -> nn.Sequential:
        layers = []
        channels = [in_channels] + list(hid_channels)
        for i in range(1, len(channels)):
            layers.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=1))
            layers.append(activation())
        if use_bn:
            layers.append(nn.BatchNorm1d(channels[-1]))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # ....................| T, 4, P
        enc_0 = self.encoder[0].forward(x)  # .......| T, 64, P
        enc_1 = self.encoder[1].forward(enc_0)  # ...| T, 1028, P
        
        pool = enc_1.max(dim=-1, keepdim=True).values
        pool = pool.expand_as(enc_1)  # .............| T, 1028, P
        
        enc = torch.cat([enc_0, pool], dim=1)  # ....| T, 1088, P
        dec = self.decoder.forward(enc)  # ..........| T, 128, P
        dec = dec.transpose(1, 2)  # ................| T, P, 128
        
        out = self.head.forward(dec)  # .............| T, P, 3
        return out
        

if __name__ == '__main__':
    model = PointNet(4, 64, 3, 8, nn.ReLU)
    sample_x = torch.randn(14, 128, 4)
    out = model.forward(sample_x)
    
    assert out.shape == (14, 128, 3)
    assert out.requires_grad
    print(PointNet, 'passed')
