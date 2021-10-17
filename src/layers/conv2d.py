import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):

    def __init__(self, ch_out, ch_in, kernel_h_sz, kernel_w_sz, stride, padding):
        super().__init__()
        self.ch_out = ch_out
        self.ch_in = ch_in
        self.kernel_h_sz = kernel_h_sz
        self.kernel_w_sz = kernel_w_sz
        self.stride = stride
        self.padding = padding

        self.w = nn.Parameter(torch.ones(self.ch_out, self.ch_in, self.kernel_h_sz, self.kernel_w_sz))
        self.b = nn.Parameter(torch.zeros(self.ch_out))

        nn.init.xavier_uniform_(self.w)


    def forward(self, x):
        x = F.conv2d(x, self.w, self.b, stride = self.stride, padding = self.padding)
        return x
        
