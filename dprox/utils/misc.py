import torch
import numpy as np

import dprox as dp

from .containar import is_dp_tensor


def to_nn_parameter(*tensor, requires_grad=False):
    """
    convert a tensor or a list of tensors to a PyTorch nn.Parameter object with an option
    to set the requires_grad attribute.
    
    :param requires_grad: requires_grad is a boolean parameter that specifies whether the tensor(s)
    should be considered for gradient computation during backpropagation. If requires_grad is set to
    True, then the tensor(s) will be included in the computation graph and their gradients will be
    computed during backpropagation. 
    :return: a `torch.nn.Parameter` object or a list of
    `torch.nn.Parameter` objects depending on the number of input tensors provided. If only one tensor
    is provided, a single `torch.nn.Parameter` object is returned. If multiple tensors are provided, a
    list of `torch.nn.Parameter` objects is returned, with each tensor converted to a
    `torch.nn.Parameter
    """
    if(len(tensor) == 1):
        return torch.nn.Parameter(tensor[0], requires_grad=requires_grad)
    else:
        return [torch.nn.Parameter(t, requires_grad=requires_grad) for t in tensor]


def batchify(out):
    """
    take an input tensor and return it as a batch tensor with an additional
    dimension.
    
    :param out: The output tensor that needs to be batchified
    :return: a PyTorch tensor. If the input tensor `out` is already a
    delta-prox tensor, it is returned as is. Otherwise, return a tensor with BCHW format 
    if the input tensor has 3 dimensions with HWC format.
    """
    if is_dp_tensor(out):
        return out
    if len(out.shape) == 3 and (out.shape[2] == 1 or out.shape[2] == 3):
        out = out.permute(2, 0, 1)
    out = out.unsqueeze(0)
    return out


def to_torch_tensor(x, batch=False):
    """
    convert a given input to a torch tensor and add a batch dimension if specified.
    
    :param x: The input data that needs to be converted to a torch tensor
    :param batch: A boolean parameter that indicates whether to add a batch dimension to the tensor or
    not. If set to True, a batch dimension will be added to the tensor, defaults to False (optional)
    :return: a torch tensor. If the input is already a dp.tensor, it returns the input as is. If the
    input is a numpy array, it converts it to a dp.tensor. If the input is neither a torch tensor nor a
    numpy array, it assumes it is a dp.tensor and returns it. If the batch parameter is True, it adds a
    batch dimension to the tensor and returns
    """
    # we don't add batch dim to dp.tensor, as it is assumed to be batched.
    if is_dp_tensor(x):
        return x

    if isinstance(x, torch.Tensor):
        out = x
    elif isinstance(x, np.ndarray):
        out = dp.tensor(x.copy())
    else:
        out = dp.tensor(x)

    if batch:
        if len(out.shape) == 3 and (out.shape[2] == 1 or out.shape[2] == 3):
            out = out.permute(2, 0, 1)
        if len(out.shape) < 4:
            out = out.unsqueeze(0)

    out.is_dp_tensor = True
    return out


def debatchify(out, squeeze):
    """
    debatchify a tensor by squeezing and/or transposing its dimensions,
    supporting multiple tensor format transforms (BCHW -> CHW | CHW -> HWC | HWC -> HW)
    
    :param out: The output tensor that needs to be transformed
    :param squeeze: A boolean parameter that determines whether to convert tensor format with C = 1 to HW format
    :return: the tensor `out` with simplified format.
    """
    if len(out.shape) == 4:
        out = out.squeeze(0)  # BCHW -> CHW
    if len(out.shape) == 3:
        if out.shape[0] == 3 or out.shape[0] == 1:
            out = out.transpose(1, 2, 0)  # CHW -> HWC
        if out.shape[2] == 1 and squeeze:
            out = out.squeeze(2)  # HWC -> HW
    return out


def to_ndarray(x, debatch=False, squeeze=False):
    """
    convert a given input into a numpy array and optionally remove any batch dimensions.
    
    :param x: The input data that needs to be converted to a numpy array
    :param debatch: A boolean parameter that specifies whether to remove the batch dimension from the
    input tensor or not. If set to True, the function will call the `debatchify` function to remove the
    batch dimension. If set to False, the function will return the input tensor as is, defaults to False
    (optional)
    :param squeeze: the `squeeze` boolean parameter in `debatchify`, that determines whether to convert 
    tensor format with C = 1 to HW format 
    :return: a numpy array. If `debatch` is True, the output is passed through the `debatchify` function
    with `squeeze` before being returned. 
    """
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        out = x.astype('float32')
    else:
        out = np.array(x)
    if debatch:
        out = debatchify(out, squeeze)
    return out


def fft2(x):
    """
    perform a 2D Fast Fourier Transform on a tensor using PyTorch.
    
    :param x: The input tensor to be transformed using 2D Fast Fourier Transform (FFT)
    :return: the 2-dimensional Fast Fourier Transform (FFT) of the input
    tensor `x`. The input tensor is first shifted using `torch.fft.ifftshift` along the last two
    dimensions (-2 and -1), then the FFT is computed using `torch.fft.fft2` with normalization set to
    'ortho', and finally the result is shifted back using `torch.fft.fft
    """
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2(x):
    """
    perform a 2D inverse fast Fourier transform on a tensor using PyTorch.
    
    :param x: The input tensor to be transformed using the inverse 2D Fast Fourier Transform (FFT)
    :return: the inverse 2D Fourier transform of the input tensor `x`.
    """
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def outlier_correct(arr, p=0.01):
    """
    replace the values in an array that fall below or above a certain
    percentile with the corresponding percentile value.
    
    :param arr: an array of numerical values
    :param p: represent the percentage of data that is considered as outliers. In this
    function, it is used to calculate the lower and upper percentiles of the data that are not
    considered as outliers. 
    :return: return the input array `arr` with its outliers corrected.
    The function replaces the values below the `p` percentile with the `p` percentile value and the
    values above the `100-p` percentile with the `100-p` percentile value.
    """
    percentile = np.percentile(arr, [p, 100-p])    
    arr[arr < percentile[0]] = percentile[0]
    arr[arr > percentile[1]] = percentile[1]
    return arr


def crop_center_region(arr, size=150):
    """
    crop a center region of a given size from a 2D array.
    
    :param arr: a numpy array representing an image
    :param size: The size of the square region to be cropped from the center of the input array. The
    default value is 150, defaults to 150 (optional)
    :return: a cropped version of the input array, where the center region of the array is extracted
    based on the specified size parameter.
    """
    # Get the dimensions of the array
    height, width = arr.shape[:2]

    # Calculate the indices for the center sizexsize region
    start_row = int((height - size) / 2)
    end_row = start_row + size
    start_col = int((width - size) / 2)
    end_col = start_col + size

    # Crop the array to the center sizexsize region
    cropped_arr = arr[start_row:end_row, start_col:end_col]

    # Return the cropped array
    return cropped_arr
