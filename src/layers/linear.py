import torch
from torch import nn
from nn import Functional as F

class Linear(nn.Module):

    def __init__(self, ch_out, ch_in):
        super().__init__()
        self.ch_out = ch_out
        self.ch_in = ch_in
        
        self.w = nn.Parameter(torch.ones(self.ch_out, self.ch_in))
        self.b = nn.Parameter(torch.zeros(ch_out))
        nn.init.xavier_uniform_(self.w)


    def forward(self, x):
        x = F.linear(x, self.w, self.b)
