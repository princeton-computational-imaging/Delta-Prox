# Absorb linear operators into proximal operators.

from dprox.linop import scale, mosaic
from dprox.proxfn import (sum_squares, weighted_sum_squares)

WEIGHTED = {sum_squares: weighted_sum_squares}


def absorb_all_linops(prox_funcs):
    """Repeatedy absorb lin ops.
    """
    new_proxes = []
    ready = prox_funcs[:]
    while len(ready) > 0:
        curr = ready.pop(0)
        absorbed = absorb_linop(curr)
        if len(absorbed) == 1 and absorbed[0] == curr:
            new_proxes.append(absorbed[0])
        else:
            ready += absorbed
    return new_proxes


def absorb_linop(prox_fn):
    """If possible moves the top level lin op argument
       into the prox operator.

       For example, elementwise multiplication can be folded into
       a separable function's prox.
    """
    if isinstance(prox_fn.linop, mosaic) and isinstance(prox_fn, sum_squares):
        new_fn = weighted_sum_squares(prox_fn.linop.input_nodes[0], prox_fn.linop, prox_fn.offset)
        return [new_fn]
    
    # Fold scalar into the function.
    if isinstance(prox_fn.linop, scale):
        scalar = prox_fn.linop.scalar
        prox_fn.linop = prox_fn.linop.input_nodes[0]
        prox_fn.beta = prox_fn.beta * scalar
        return [prox_fn]
    # No change.
    return [prox_fn]
