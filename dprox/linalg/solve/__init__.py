from .cg import conjugate_gradient, conjugate_gradient2, preconditioned_conjugate_gradient
from .plss import PLSS, PLSSW
from .minres import MINRES


__all__ = available_solvers = [
    'conjugate_gradient',
    'conjugate_gradient2',
    'preconditioned_conjugate_gradient',
    'PLSS',
    'PLSSW',
    'MINRES',
]

SOLVERS = {
    'cg': conjugate_gradient,
    'cg2': conjugate_gradient2,
    'pcg': preconditioned_conjugate_gradient,
    'plss': PLSS,
    'plssw': PLSSW,
    'minres': MINRES,
}
