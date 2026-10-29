import torch
from torch import nn

class DropPath(nn.Module):
    def __init__(self,drop_prob=0.):
        super().__init__()
        self.drop_prob=drop_prob
    def forward(self,x):
        if not self.training or not self.drop_prob:
            return x
        keep=1-self.drop_prob
        mask=x.new_empty((x.shape[0],)+(1,)*(x.ndim-1)).bernoulli_(keep)
        return x*mask/keep

trunc_normal_=nn.init.trunc_normal_

def load_state_dict(model,state_dict,strict=False):
    return model.load_state_dict(state_dict,strict=strict)
