import numpy as np
import torch


def is_dp_array(x):
    return hasattr(x, 'is_dp_array') and x.is_dp_array == True


def array(*args, **kwargs):
    out = np.array(*args, **kwargs)
    out.is_dp_array = True
    return out


def is_dp_tensor(x):
    return hasattr(x, 'is_dp_tensor') and x.is_dp_tensor == True


def tensor(*args, **kwargs):
    out = torch.tensor(*args, **kwargs)
    out.is_dp_tensor = True
    return out
