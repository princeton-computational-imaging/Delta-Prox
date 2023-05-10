from .cg import conjugate_gradient, conjugate_gradient2

__all__ = available_solvers = [
    'conjugate_gradient',
    'conjugate_gradient2'
]

SOLVERS = {
    'cg': conjugate_gradient,
    'cg2': conjugate_gradient2,
}
