import torch

from .base import LinOp


class sum(LinOp):
    """Sums its inputs.
    """

    def __init__(self, input_nodes):
        super(sum, self).__init__(input_nodes)

    def forward(self, *inputs, **kwargs):
        """ Just sum all the inputs, all inputs should have the same shape
        """
        output = torch.zeros_like(inputs[0])
        for input in inputs:
            output += input.to(output.device)
        return output

    def adjoint(self, input, **kwargs):
        """ The adjoint of sum spread of the input to all its child
        """
        outputs = LinOp.MultOutput()
        for _ in self.input_nodes:
            outputs.append(input)
        if len(outputs) > 1:
            return outputs
        return outputs[0]
    
    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return all([arg.is_diag(freq) for arg in self.input_nodes])

    def is_gram_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return all([arg.is_gram_diag(freq) for arg in self.input_nodes])

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
        # var_diags = {var: torch.zeros(var.size) for var in self.variables()}
        # for arg in self.input_nodes:
        #     arg_diags = arg.get_diag(shape, freq)
        #     for var, diag in arg_diags.items():
        #         var_diags[var] = var_diags[var] + diag
        # return var_diags.values()[0]
        return self.input_nodes[0].get_diag(ref, freq)

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
        return torch.sum(input_mags)


class copy(sum):

    def __init__(self, arg):
        super(copy, self).__init__([arg])

    def forward(self, inputs, **kwargs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        return super(copy, self).adjoint(inputs, **kwargs)

    def adjoint(self, *inputs, **kwargs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return super(copy, self).forward(*inputs, **kwargs)

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
