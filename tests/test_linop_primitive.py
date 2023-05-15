import dprox as dp
import torch

def test_eval():
    op = dp.Variable()
    out = dp.eval(op, torch.zeros(64,64))
    print(out)
    assert torch.allclose(out, torch.zeros(64,64))