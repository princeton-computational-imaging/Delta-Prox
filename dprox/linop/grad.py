import numpy as np

from dprox.utils.misc import to_torch_tensor

from .conv import conv


class grad(conv):
    """
    gradient operation. can be defined for different dimensions.
    default is n-d gradient.
    """

    def __init__(self, arg, dim=1):
        if dim not in [0,1,2]:
            raise ValueError('dim must be 0(Height) or 1(Width) or 2 (Channel)')
        
        D = to_torch_tensor([1, -1])
        for _ in range(3-1):
            D = D.unsqueeze(0)
        D = D.transpose(dim, -1)
        
        super(grad, self).__init__(arg, kernel=D)

    def get_dims(self):
        """Return the dimensinonality of the gradient
        """
        return self.dims

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
        # 1D gradient operator has spectral norm = 2.
        # ND gradient is permutation of stacked grad in axis 0, axis 1, etc.
        # so norm is 2*sqrt(dims)
        return 2 * np.sqrt(self.dims) * input_mags[0]
