import numpy as np
import torch

from .base import LinOp


class scale(LinOp):
    """Multiplication scale*X with a fixed scalar.
    """

    def __init__(self, scalar, arg):
        assert np.isscalar(scalar)
        self.scalar = scalar
        super(scale, self).__init__([arg])

    # ---------------------------------------------------------------------------- #
    #                                  Computation                                 #
    # ---------------------------------------------------------------------------- #

    def forward(self, input, **kwargs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        return input * self.scalar

    def adjoint(self, input, **kwargs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return self.forward(input)

    # ---------------------------------------------------------------------------- #
    #                                   Diagonal                                   #
    # ---------------------------------------------------------------------------- #

    def is_gram_diag(self, freq=False):
        """Is the lin  Gram diagonal (in the frequency domain)?
        """
        return self.input_nodes[0].is_gram_diag(freq)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return self.input_nodes[0].is_diag(freq)

    def get_diag(self, ref, freq=False):
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
        var_diags = self.input_nodes[0].get_diag(ref, freq) * self.scalar
        return var_diags * torch.conj(var_diags)

    # ---------------------------------------------------------------------------- #
    #                                   Property                                   #
    # ---------------------------------------------------------------------------- #

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
        return abs(self.scalar) * input_mags[0]
