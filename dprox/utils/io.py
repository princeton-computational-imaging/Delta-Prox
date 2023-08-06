import os
from typing import List

import imageio
import numpy as np
import matplotlib.pyplot as plt

from .misc import to_ndarray


def imread_rgb(path: str) -> np.ndarray:
    """
    Read an image file from a given path and return it as a NumPy array with RGB color channels.

    Args:
      path (str): The path to the image file that needs to be read

    Return:
      A NumPy array representing the RGB image read from the specified file path.
    """
    import numpy as np
    from PIL import Image
    img = Image.open(path)
    return np.asarray(img)


def imshow(*imgs: List[np.ndarray],
           maxcol: int = 3,
           gray: bool = False,
           titles: List[str] = None,
           off_axis: bool = True) -> None:
    """
    Display one or more images in a grid with customizable parameters such as 
    maximum number of columns, grayscale, and titles.

    Args:
      imgs (List[np.ndarray]): a list of images.
      maxcol (int): The maximum number of columns to display the images in. If there are more images than
        maxcol, they will be displayed in multiple rows. The default value is 3, defaults to 3 (optional)
      gray (bool): A boolean parameter that determines whether the image(s) should be displayed in
        grayscale or in color. If set to True, the images will be displayed in grayscale. If set to False,
        the images will be displayed in color, defaults to False (optional)
      titles (List[str]): titles is a list of strings that contains the titles for each image being displayed. If titles is None, then no titles will be displayed
      off_axis (bool): whether to remove axis in the images.
    """
    if len(imgs) != 1:
        plt.figure(figsize=(10, 5), dpi=300)
    row = (len(imgs) - 1) // maxcol + 1
    col = maxcol if len(imgs) >= maxcol else len(imgs)
    for idx, img in enumerate(imgs):
        img = to_ndarray(img, debatch=True)
        if img.max() > 2: img = img / 255
        img = img.clip(0, 1)
        if gray: plt.gray()
        plt.subplot(row, col, idx + 1)
        plt.imshow(img)
        if titles is not None: plt.title(titles[idx])
        if off_axis: plt.axis('off')
    plt.show()


def imread(path: str) -> np.ndarray:
    """
    read an image from a given path and return it as a numpy array of float values
    between 0 and 1.

    Args:
      path (str): a string representing the file path of an image file to be read

    Return: 
      a NumPy array of type `float32` representing an image that has been read from the file path specified in the input argument. The pixel values of the image are normalized to the range [0, 1] by dividing each pixel value by 255.
    """
    img = imageio.imread(path)
    return np.float32(img) / 255


def filter_ckpt(prefix: str, ckpt: dict, remove_prefix: bool = True):
    """
    filter a checkpoint dictionary by a given prefix and optionally remove the prefix from the keys.

    Args:
      prefix (str): The prefix is a string that is used to filter the keys of the checkpoint dictionary. 
        Only the keys that start with this prefix will be included in the new checkpoint dictionary
      ckpt (dict): The ckpt parameter is a dictionary containing the keys and values of a TensorFlow
        checkpoint file. It typically contains the weights and biases of a trained model
      remove_prefix (bool): remove_prefix is a boolean parameter that determines whether the prefix should
        be removed from the keys of the returned dictionary. If set to True, the prefix will be removed,
        otherwise, the keys will remain unchanged, defaults to True (optional)

    Return: 
      a new dictionary `new_ckpt` that contains the same values as the input dictionary `ckpt`,
      but with the keys that start with the `prefix` string removed (if `remove_prefix` is True) or
      unchanged (if `remove_prefix` is False).
    """
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith(prefix):
            if remove_prefix: new_k = k.replace(prefix, '')
            else: new_k = k
            new_ckpt[new_k] = v
    return new_ckpt


def is_image_file(filename):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # Check if the file extension is in the image extensions list (case insensitive)
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def list_image_files(directory):
    image_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and is_image_file(filename):
            image_files.append(filename)
    return image_files
