import abc

import numpy as np
import torch
import torch.nn as nn

from dprox.utils import to_torch_tensor


def cast_to_const(expr):
    """Converts a non-LinOp to a Constant.
    """
    from .constant import Constant
    return expr if isinstance(expr, LinOp) else Constant(expr)


class LinOp(nn.Module):
    """Represents a linear operator.
    """
    __metaclass__ = abc.ABCMeta

    instanceCnt = 0

    def __init__(self, input_nodes):
        super(LinOp, self).__init__()
        self.input_nodes = nn.ModuleList([cast_to_const(node) for node in input_nodes])
        self.linop_id = LinOp.instanceCnt
        LinOp.instanceCnt += 1
        self.dummy = torch.nn.parameter.Parameter(torch.tensor(0), requires_grad=False)

        # will be set later by proximal algorithm to indicate current iteration step
        self.step = 0

    @property
    def device(self):
        return self.dummy.device

    def unwrap(self, value):
        from .placeholder import Placeholder
        if isinstance(value, Placeholder):
            return value.value
        return to_torch_tensor(value, batch=True)

    def params(self):
        params = []
        for node in self.input_nodes:
            params += node.params()
        return params
    
    def variables(self):
        """Return the list of variables used in the LinOp.
        """
        vars_ = []
        for arg in self.input_nodes:
            vars_ += arg.variables()
        unordered = list(set(vars_))  # Make unique, order by uuid.
        return sorted(unordered, key=lambda x: x.uuid)

    def constants(self):
        """Returns a list of constants in the LinOp.
        """
        consts = []
        for arg in self.input_nodes:
            consts += arg.constants()
        return consts

    def is_constant(self):
        """Is the LinOp constant?
        """
        return len(self.variables()) == 0

    @abc.abstractmethod
    def forward(self, inputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        return NotImplemented

    @abc.abstractmethod
    def adjoint(self, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return NotImplemented

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return self.is_diag(freq)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return False

    def get_diag(self, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        return NotImplemented

    def __add__(self, other):
        """Lin Op + Lin Op.
        """
        other = cast_to_const(other)
        from .sum import sum
        args = []
        for elem in [self, other]:
            if isinstance(elem, sum):
                args += elem.input_nodes
            else:
                args += [elem]
        return sum(args)

    def __mul__(self, other):
        """Lin Op * Number.
        """
        from .scale import scale

        # Can only divide by scalar constants.
        if np.isscalar(other):
            return scale(other, self)
        else:
            raise TypeError("Can only multiply by a scalar constant.")

    def __rmul__(self, other):
        """Called for Number * Lin Op.
        """
        return self * other

    def __truediv__(self, other):
        """Lin Op / Number.
        """
        return self.__div__(other)

    def __div__(self, other):
        """Lin Op / Number.
        """
        from .scale import scale

        # Can only divide by scalar constants.
        if np.isscalar(other):
            return scale(1. / other, self)
        else:
            raise TypeError("Can only divide by a scalar constant.")

    def __sub__(self, other):
        """Called for lin op - other.
        """
        return self + -other

    def __rsub__(self, other):
        """Called for other - lin_op.
        """
        return -self + other

    def __neg__(self):
        """The negation of the Lin Op.
        """
        return -1 * self

    def __str__(self):
        """Default to string is name of class.
        """
        return self.__class__.__name__

    @property
    def value(self):
        inputs = []
        for node in self.input_nodes:
            inputs.append(node.value)
        output = self.forward(inputs)[0]
        return output

    def get_offset(self):
        """Get the constant offset.
        """
        old_vals = {}
        for var in self.variables():
            old_vals[var] = var.value
            var.value = torch.zeros_like(var.value)
        offset = self.value
        # Restore old variable values.
        for var in self.variables():
            var.value = old_vals[var]
        return offset

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        return NotImplemented