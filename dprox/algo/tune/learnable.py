import torch
import torch.nn as nn


class LearnableParamProvider(nn.Module):
    def __init__(self, steps, default_value=0.5):
        super().__init__()
        # for step in steps:
        #     self.register_buffer(f'')
        # self.params = nn.Parameter(torch.ones([len(steps)]) * default_value)

    def forward(self, step):
        # return self.params[step]
        if step == 0: return torch.tensor(0.3442)
        elif step == 6: return torch.tensor(0.6111)
        else: return torch.tensor(0.3168)