from .solver_cg import cg, cg2, pcg
from .solver_plss import plss, plssw
from .solver_minres import minres


__all__ = available_solvers = [
    'cg',
    'cg2',
    'pcg',
    'plss',
    'plssw',
    'minres',
]

SOLVERS = {
    'cg': cg,
    'cg2': cg2,
    'pcg': pcg,
    'plss': plss,
    'plssw': plssw,
    'minres': minres,
}
