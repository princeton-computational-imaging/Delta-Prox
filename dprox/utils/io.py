import os
import urllib.request

import imageio
import numpy as np
from tqdm import tqdm

from . import to_ndarray


def imread_rgb(path):
    """
    Read an image from a file.
    """
    import numpy as np
    from PIL import Image
    img = Image.open(path)
    return np.asarray(img)


def imshow(*imgs, maxcol=3, gray=False):
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
    plt.show()


def imread(path):
    img = imageio.imread(path)
    return np.float32(img) / 255


def filter_ckpt(prefix, ckpt, remove_prefix=True):
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith(prefix):
            if remove_prefix: new_k = k.replace(prefix, '')
            else: new_k = k
            new_ckpt[new_k] = v
    return new_ckpt


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_path(base_path):
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


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
