from .cg import conjugate_gradient, conjugate_gradient2, conjugate_gradient3
from .custom import custom_conjugate_gradient, custom_conjugate_gradient2
from .torchcg import batch_conjugate_gradient

LINEAR_SOLVER = {
    'cg': conjugate_gradient,
    'cg2': conjugate_gradient2,
    'cg3': conjugate_gradient3,
    'custom': custom_conjugate_gradient,
    'custom2': custom_conjugate_gradient2,
    'bcg': batch_conjugate_gradient
}
