import torch
import numpy as np


def to_nn_parameter(*tensor, requires_grad=False):
    if(len(tensor) == 1):
        return torch.nn.Parameter(tensor[0], requires_grad=requires_grad)
    else:
        return [torch.nn.Parameter(t, requires_grad=requires_grad) for t in tensor]


def batchify(out):
    if len(out.shape) == 2:
        out = out.unsqueeze(-1)
    if len(out.shape) == 3 and out.shape[2] == 1 or out.shape[2] == 3:
        out = out.permute(2, 0, 1)
    if len(out.shape) == 3:
        out = out.unsqueeze(0)
    return out


def to_torch_tensor(x, batch=False):
    if isinstance(x, torch.Tensor):
        out = x
    elif isinstance(x, np.ndarray):
        out = torch.from_numpy(x.copy())
    else:
        out = torch.tensor(x)

    if batch:
        out = batchify(out)
    return out


def debatchify(out, squeeze):
    if len(out.shape) == 4:
        out = out.squeeze(0)  # NCHW -> CHW
    if len(out.shape) == 3:
        if out.shape[0] == 3 or out.shape[0] == 1:
            out = out.transpose(1, 2, 0)  # CHW -> HWC
        if out.shape[2] == 1 and squeeze:
            out = out.squeeze(2)
    return out


def to_ndarray(x, debatch=False, squeeze=False):
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        out = x.astype('float32')
    else:
        out = np.array(x)
    if debatch:
        out = debatchify(out, squeeze)
    return out
