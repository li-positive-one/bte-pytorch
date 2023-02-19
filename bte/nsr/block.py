import torch
import torch.nn as nn

class FCSeq(nn.Module):
    def __init__(self, neurons, act=torch.sin):
        super().__init__()
        assert len(neurons) >= 1
        self.linears = nn.ModuleList(
            [nn.Linear(neurons[i], neurons[i + 1]) for i in range(0, len(neurons) - 1)]
        )
        self.act = act

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = l(x)
            if i != len(self.linears) - 1:
                x = self.act(x)
        return x
    
class MultiResFCSeq(nn.Module):
    def __init__(self, neurons, act=torch.sin, multires=(1,4,16)):
        super().__init__()
        self.multires=multires
        self.net = FCSeq([len(self.multires)*neurons[0],]+neurons[1:],act=act)
        
    def forward(self, x):
        xs = torch.cat([s*x for s in self.multires], dim=-1)
        y = self.net(xs)
        return y