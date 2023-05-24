import numpy as np
import torch

from .base import LinOp


class Constant(LinOp):
    """A constant.
    """

    def __init__(self, value):
        super(Constant, self).__init__([])
        if value is not None and not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self._value = value

    # ---------------------------------------------------------------------------- #
    #                                  Computation                                 #
    # ---------------------------------------------------------------------------- #

    def forward(self, *value, **kwargs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        return self.value

    def adjoint(self, value, **kwargs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return self.value*0

    # ---------------------------------------------------------------------------- #
    #                                   Diagonal                                   #
    # ---------------------------------------------------------------------------- #

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return True

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
        return {}

    # ---------------------------------------------------------------------------- #
    #                                   Property                                   #
    # ---------------------------------------------------------------------------- #

    @property
    def variables(self):
        return []

    @property
    def constants(self):
        return [self]
    
    @property
    def value(self):
        return self._value.to(self.device)
    
    
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
        return 0.0

    # ---------------------------------------------------------------------------- #
    #                                 Python Magic                                 #
    # ---------------------------------------------------------------------------- #
    
    def __repr__(self):
        if self._value is not None:
            return 'Constant(value=somevalue)'
        return 'Constant(value=None)'