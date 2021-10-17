import torch
from torch import nn
from torch.nn import functional as F

class Bn(nn.Module):

    def __init__(self, ch_out):
        super().__init__()
        self.ch_out = ch_out
        
        self.w = nn.Parameter(torch.ones(self.ch_out))
        self.b = nn.Parameter(torch.zeros(self.ch_out))
        self.running_mean = nn.Parameter(torch.zeros(ch_out), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(ch_out), requires_grad=False)


    def forward(self, x, bn_training = True):
        x = F.batch_norm(x, self.running_mean, self.running_var, weight = self.w, bias = self.b, training = bn_training)
