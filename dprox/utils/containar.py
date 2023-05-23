import numpy as np
import torch


def is_dp_array(x):
    """
    check if an object has an attribute 'is_dp_array' with a value of True.
    
    :param x: The input parameter to the function is x, which is expected to be an object
    :return: a boolean value. It checks if the input `x` has an
    attribute called `is_dp_array` and if its value is `True`.
    """
    return hasattr(x, 'is_dp_array') and x.is_dp_array == True


def array(*args, **kwargs):
    """
    create a numpy array and sets an attribute to indicate that it is a dp_array.
    :return: a numpy array with an additional attribute `is_dp_array`
    set to `True`.
    """
    out = np.array(*args, **kwargs)
    out.is_dp_array = True
    return out


def is_dp_tensor(x):
    """
    check if the input object has an attribute 'is_dp_tensor' set to True, indicating that
    it is a delta-prox tensor.
    
    :param x: The input variable that is being checked for whether it is a delta-prox tensor
    or not
    :return: a boolean value `True` if the input `x` has an
    attribute `is_dp_tensor` that is also `True`, and `False` otherwise.
    """
    return hasattr(x, 'is_dp_tensor') and x.is_dp_tensor == True


def tensor(*args, **kwargs):
    """
    create a PyTorch tensor and set a flag to indicate that it is a
    delta-prox tensor.
    :return: A PyTorch tensor with an additional attribute `is_dp_tensor` set to `True`.
    """
    out = torch.tensor(*args, **kwargs)
    out.is_dp_tensor = True
    return out
