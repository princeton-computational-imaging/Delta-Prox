# Implementation of matlab psf2otf function.
# Adapted from matlab psf2otf source code:
# https://github.com/david-hoffman/pydecon/blob/master/notebooks/psf2otf.m
#
# See also:
# https://blog.csdn.net/bluecol/article/details/48288739

import numpy as np


def psf2otf(psf, outsize):
    """
    convert a point spread function (PSF) to an optical transfer function (OTF)
    using fast Fourier transform (FFT).
    
    :param psf: a 2D or 3D array representing the point spread function
    :param outsize: The desired output size of the optical transfer function (OTF)
    :return: the optical transfer function (OTF) of a point spread function (PSF) after padding and
    circularly shifting it.
    """
    psf, psfsize, outsize = _process_args(psf, outsize)
    if not np.all(psf == 0):
        padsize = outsize - psfsize
        psf = padarray(psf, padsize, 'post')

        # Circularly shift otf so that the "center" of the PSF is at the
        # (1,1) element of the array    .
        psf = circshift(psf, -np.floor(psfsize/2).astype(np.int32))
        otf = np.fft.fftn(psf)

        # Estimate the rough number of operations involved in the FFT
        # and discard the PSF imaginary part if within roundoff error
        # roundoff error = machine epsilon = sys.float_info.epsilon
        # or np.finfo().eps
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)

    else:
        otf = np.zeros(outsize)
    return otf


def _process_args(psf, outsize):
    psfsize = np.array(psf.shape)
    outsize = np.array(outsize)
    if len(psfsize) > len(outsize):
        raise ValueError('psf must be smaller than outsize')
    for _ in range(len(outsize) - len(psfsize)):
        psf = np.expand_dims(psf, axis=-1)
    psfsize = np.pad(psfsize, (0, len(outsize) - len(psfsize)),
                     'constant', constant_values=1)
    if any(psfsize > outsize):
        raise ValueError('outsize {} cannot be smaller than the PSF array size {} in any dimension.'.format(outsize, psfsize))
    return psf, psfsize, outsize


def padarray(array, padsize, direction='post'):
    """
    pad an array with zeros in a specified direction and size.
    
    :param array: The input array to be padded
    :param padsize: The amount of padding to add to the array in each dimension
    :param direction: The direction parameter specifies where the padding should be added in relation to
    the input array. It can take three possible values: 'post' (padding is added at the end of the
    array), 'pre' (padding is added at the beginning of the array), or 'both' (padding is added,
    defaults to post (optional)
    :return: a numpy array with padding added to the input array based on the specified padsize and
    direction.
    """
    pad_width = []
    for size in padsize:
        if direction == 'post':
            pad_width.append((0, size))
        elif direction == 'pre':
            pad_width.append((size, 0))
        elif direction == 'both':
            pad_width.append((size, size))
        else:
            raise ValueError('direction must be "both" or "post" or "pre"')
    out = np.pad(array, pad_width=pad_width, mode='constant', constant_values=0)
    return out


def circshift(array, shift):
    """
    shift the elements of a numpy array along one or more axes by a specified
    number of positions.
    
    :param array: The input array that needs to be circularly shifted
    :param shift: shift is a list of integers that specifies the number of positions by which the
    elements of the input array should be shifted along each axis. The length of the shift list should
    be equal to the number of dimensions of the input array
    :return: The function `circshift` returns the input array after performing circular shifts along
    each axis specified by the `shift` parameter.
    """
    for axis, s in enumerate(shift):
        array = np.roll(array, s, axis=axis)
    return array
