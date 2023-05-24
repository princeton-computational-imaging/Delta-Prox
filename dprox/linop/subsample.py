import numpy as np

from dprox.utils.misc import to_nn_parameter, to_torch_tensor

from .base import LinOp


class mosaic(LinOp):

    def __init__(self, arg):
        super(mosaic, self).__init__([arg])
        self.cache = {}

    # ---------------------------------------------------------------------------- #
    #                                  Computation                                 #
    # ---------------------------------------------------------------------------- #

    def forward(self, input, **kwargs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        mask = self._mask(input.shape).to(input.device)
        return mask * input

    def adjoint(self, input, **kwargs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        return self.forward(input)

    @staticmethod
    def masks_CFA_Bayer(shape):
        pattern = 'RGGB'
        channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
        for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            channels[channel][y::2, x::2] = 1
        return tuple(channels[c].astype(bool) for c in 'RGB')

    def _mask(self, shape):
        if shape not in self.cache:
            shape = shape[-2:]
            R_m, G_m, B_m = self.masks_CFA_Bayer(shape)
            mask = np.concatenate((R_m[..., None], G_m[..., None], B_m[..., None]), axis=-1)
            self.cache[shape] = to_nn_parameter(to_torch_tensor(mask.astype('float32'), batch=True))
        return self.cache[shape]

    # ---------------------------------------------------------------------------- #
    #                                   Diagonal                                   #
    # ---------------------------------------------------------------------------- #

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return self.is_self_diag(freq) and self.input_nodes[0].is_diag(freq)

    def is_self_diag(self, freq=False):
        return not freq

    def get_diag(self, x, freq=False):
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
        assert not freq
        # var_diags = self.input_nodes[0].get_diag(freq)
        # selection = self.get_selection()
        # self_diag = np.zeros(self.input_nodes[0].shape)
        # self_diag[selection] = 1
        # for var in var_diags.keys():
        #     var_diags[var] = var_diags[var] * self_diag.ravel()
        return self._mask(x.shape).to(self.device)

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
        return input_mags[0]
