import os
import urllib.request

import imageio
import numpy as np
from tqdm import tqdm

from . import to_ndarray


def imread_rgb(path) -> np.ndarray:
    """
    read an image file from a given path and return it as a NumPy array with RGB color
    channels.
    
    :param path: The path to the image file that needs to be read
    :return: a NumPy array representing the RGB image read from the
    specified file path.
    """
    import numpy as np
    from PIL import Image
    img = Image.open(path)
    return np.asarray(img)


def imshow(*imgs, maxcol=3, gray=False, titles=None) -> None:
    """
    display one or more images in a grid with customizable parameters such as
    maximum number of columns, grayscale, and titles.
    
    :param maxcol: The maximum number of columns to display the images in. If there are more images than
    maxcol, they will be displayed in multiple rows. The default value is 3, defaults to 3 (optional)
    :param gray: A boolean parameter that determines whether the image(s) should be displayed in
    grayscale or in color. If set to True, the images will be displayed in grayscale. If set to False,
    the images will be displayed in color, defaults to False (optional)
    :param titles: titles is a list of strings that contains the titles for each image being displayed.
    If titles is None, then no titles will be displayed
    """
    import matplotlib.pyplot as plt
    if len(imgs) != 1:
        plt.figure(figsize=(10, 5))
    row = (len(imgs) - 1) // maxcol + 1
    col = maxcol if len(imgs) >= maxcol else len(imgs)
    for idx, img in enumerate(imgs):
        img = to_ndarray(img, debatch=True)
        if img.max() > 2: img = img / 255
        img = img.clip(0, 1)
        if gray or len(img.shape) == 2: plt.gray()
        plt.subplot(row, col, idx + 1)
        plt.imshow(img)
        if titles is not None: plt.title(titles[idx])
    plt.show()


def imread(path) -> np.ndarray:
    """
    read an image from a given path and return it as a numpy array of float values
    between 0 and 1.
    
    :param path: a string representing the file path of an image file to be read
    :return: a NumPy array of type `float32` representing an image that
    has been read from the file path specified in the input argument. The pixel values of the image are
    normalized to the range [0, 1] by dividing each pixel value by 255.
    """
    img = imageio.imread(get_path(path))
    return np.float32(img) / 255


def filter_ckpt(prefix, ckpt, remove_prefix=True):
    """
    filter a checkpoint dictionary by a given prefix and optionally remove the prefix
    from the keys.
    
    :param prefix: The prefix is a string that is used to filter the keys of the checkpoint dictionary.
    Only the keys that start with this prefix will be included in the new checkpoint dictionary
    :param ckpt: The ckpt parameter is a dictionary containing the keys and values of a TensorFlow
    checkpoint file. It typically contains the weights and biases of a trained model
    :param remove_prefix: remove_prefix is a boolean parameter that determines whether the prefix should
    be removed from the keys of the returned dictionary. If set to True, the prefix will be removed,
    otherwise, the keys will remain unchanged, defaults to True (optional)
    :return: a new dictionary `new_ckpt` that contains the same values as the input dictionary `ckpt`,
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


# The DownloadProgressBar class is a subclass of the tqdm class in Python used to display progress
# bars for downloading files.
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_path(base_path) -> str:
    """
    check if a file exists in a specific directory and download it from a URL if it
    doesn't exist.
    
    :param base_path: The base path is a string that represents the path to a file or directory that the
    function is trying to locate or download. It is used to construct the full path to the file or
    directory by appending it to the DPROX_DIR path
    :return: a string which is the path to the file specified by the input parameter `base_path`.
    """
    DPROX_DIR = os.path.join(os.path.expanduser('~'), '.cache/dprox')

    save_path = os.path.join(DPROX_DIR, base_path)
    if not os.path.exists(save_path):
        url = f"https://huggingface.co/aaronb/DeltaProx/resolve/main/{base_path}"
        print(f'{base_path} not found')
        print('Try to download from huggingface: ', url)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        download_url(url, save_path)
        print('Downloaded to ', save_path)
    return save_path


def download_url(url, output_path) -> None:
    """
    download a file from a given URL and save it to a specified output path while
    displaying a progress bar.
    
    :param url: The URL of the file to be downloaded
    :param output_path: output_path is a string representing the file path where the downloaded file
    will be saved. It should include the file name and extension
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
