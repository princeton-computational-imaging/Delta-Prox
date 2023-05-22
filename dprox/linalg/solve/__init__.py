from .cg import conjugate_gradient, conjugate_gradient2, PCG
from .plss import PLSS, PLSSW
from .minres import MINRES


__all__ = available_solvers = [
    'conjugate_gradient',
    'conjugate_gradient2',
    'PCG',
    'PLSS',
    'PLSSW',
    'MINRES',
]

SOLVERS = {
    'cg': conjugate_gradient,
    'cg2': conjugate_gradient2,
    'pcg': PCG,
    'plss': PLSS,
    'plssw': PLSSW,
    'minres': MINRES,
}
