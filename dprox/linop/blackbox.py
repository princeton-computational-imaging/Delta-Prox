from .base import LinOp


def LinOpFactory(forward, adjoint, diag=None, norm_bound=None):
    """Returns a function to generate a custom LinOp.

    Parameters
    ----------
    input_shape : tuple
        The dimensions of the input.
    output_shape : tuple
        The dimensions of the output.
    forward : function
        Applies the operator to an input array and writes to an output.
    adjoint : function
        Applies the adjoint operator to an input array and writes to an output.
    norm_bound : float, optional
        An upper bound on the spectral norm of the operator.
    """
    def get_black_box(*args):
        return BlackBox(*args, forward=forward, adjoint=adjoint, diag=diag, norm_bound=norm_bound)
    return get_black_box


class BlackBox(LinOp):
    """A black-box lin op specified by the user.
    """

    def __init__(self, *args, forward=None, adjoint=None, diag=None, norm_bound=None):
        self._forward = forward
        self._adjoint = adjoint
        self._norm_bound = norm_bound
        self._diag = diag
        super(BlackBox, self).__init__(args)

    def forward(self, *inputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        if len(inputs) == 1:
            return self._forward(inputs[0], step=self.step)
        return self._forward(*inputs, step=self.step)

    def adjoint(self, *inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        if len(inputs) == 1:
            return self._adjoint(inputs[0], step=self.step)
        return self._adjoint(*inputs, step=self.step)

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
        if self._norm_bound is None:
            return super(BlackBox, self).norm_bound(input_mags)
        else:
            return self._norm_bound * input_mags[0]

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return self._diag != None

    def get_diag(self, x, freq=False):
        return self._diag(x, self.step)
