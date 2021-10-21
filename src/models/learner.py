import torch
from torch import nn
from src.layers.bn import Bn
from src.layers.conv2d import Conv2d
from src.layers.linear import Linear

class Learner(nn.module):
    def __init__(self, num_filters, n_way, kernels_h_sz = 3, kernels_w_sz = 3, strides = 2, paddings = 1):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_h_sz = kernels_h_sz
        self.kernel_w_sz = kernels_w_sz
        self.stride = strides
        self.padding = paddings
        self.n_way = n_way

        self.net = nn.Sequential(
            
            Conv2d(self.num_filters,1,self.kernel_h_sz,self.kernel_w_sz,self.stride,self.padding),
            nn.ReLu(),
            Bn(self.num_filters),

            Conv2d(self.num_filters,self.num_filters,self.kernel_h_sz,self.kernel_w_sz,self.stride,self.padding),
            nn.ReLu(),
            Bn(self.num_filters),

            Conv2d(self.num_filters,self.num_filters,self.kernel_h_sz,self.kernel_w_sz,self.stride,self.padding),
            nn.ReLu(),
            Bn(self.num_filters),

            Conv2d(self.num_filters,self.num_filters,self.kernel_h_sz,self.kernel_w_sz,self.stride,self.padding),
            nn.ReLu(),
            Bn(self.num_filters),
            nn.Flatten(),

            Linear(self.n_way+2, self.num_filters*9)
        )

    def forward(self,x):
        return self.net(x)
