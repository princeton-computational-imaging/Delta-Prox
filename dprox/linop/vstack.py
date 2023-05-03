import numpy as np

from .lin_op import LinOp


class vstack(LinOp):
    """Vectorizes and stacks inputs.
    """

    def __init__(self, input_nodes):
        super(vstack, self).__init__(input_nodes)

    def forward(self, inputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        return inputs

    def adjoint(self, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return inputs

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return all([arg.is_gram_diag(freq) for arg in self.input_nodes])

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
        var_diags = {var: np.zeros(var.size) for var in self.variables()}
        for arg in self.input_nodes:
            arg_diags = arg.get_diag(freq)
            for var, diag in arg_diags.items():
                var_diags[var] = var_diags[var] + diag * np.conj(diag)
        # Get (A^TA)^{1/2}
        for var in self.variables():
            var_diags[var] = np.sqrt(var_diags[var])
        return var_diags

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
        return np.linalg.norm(input_mags, 2)


class split(vstack):

    def __init__(self, output_nodes):
        self.output_nodes = output_nodes
        self.input_nodes = []
        super(split, self).__init__(output_nodes)

    def forward(self, inputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        return super(split, self).adjoint(inputs)

    def adjoint(self, inputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return super(split, self).forward(inputs)

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
        return input_mags[0]