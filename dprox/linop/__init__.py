from .blackbox import LinOpFactory, BlackBox
from .conv import conv, conv_doe
from .constant import Constant
from .comp_graph import CompGraph, est_CompGraph_norm, eval, adjoint, gram, validate
from .scale import scale
from .subsample import mosaic
from .sum import sum, copy
from .variable import Variable
from .base import LinOp
from .vstack import vstack, split
from .placeholder import Placeholder
from .grad import grad
from .mul import mul_color, mul_elementwise