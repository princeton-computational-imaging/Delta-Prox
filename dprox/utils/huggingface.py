import os
import urllib.request
import huggingface_hub

from tqdm import tqdm

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache/dprox')


class DownloadProgressBar(tqdm):
    # The DownloadProgressBar class is a subclass of the tqdm class in Python used to
    # display progress bars for downloading files.
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str) -> None:
    """
    download a file from a given URL and save it to a specified output path while
    displaying a progress bar.

    Args:
      url (str): The URL of the file to be downloaded
      output_path (str): output_path is a string representing the file path where the downloaded file
        will be saved. It should include the file name and extension
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def load_path(base_path: str, repo_type='datasets', user_id='delta-prox') -> str:
    """
    check if a file exists in a specific directory and download it from a URL if it
    doesn't exist.

    Args:
      base_path (str): The base path is a string that represents the path to a file or directory that the
        function is trying to locate or download. It is used to construct the full path to the file or
        directory by appending it to the DPROX_DIR path

    Return: 
      a string which is the path to the file specified by the input parameter `base_path`.
    """
    if os.path.exists(base_path):
        return base_path

    save_path = os.path.join(CACHE_DIR, base_path)
    if not os.path.exists(save_path):
        base_url = 'https://huggingface.co'
        if repo_type == 'datasets':
            base_url += '/' + repo_type
        repo_id = base_path.split('/')[0]
        path = os.path.join(*(base_path.split('/')[1:]))
        url = f"{base_url}/{user_id}/{repo_id}/resolve/main/{path}"
        print(f'{base_path} not found')
        print('Try to download from huggingface: ', url)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        download_url(url, save_path)
        print('Downloaded to ', save_path)
    return save_path


def load_image(path, user_id='delta-prox'):
    import imageio
    import numpy as np
    path = load_path(path, user_id=user_id)
    img = imageio.imread(path)
    return np.float32(img) / 255


def load_checkpoint(path, user_id='delta-prox'):
    import torch
    ckpt_path = load_path(path, repo_type='models', user_id=user_id)
    return torch.load(ckpt_path)


def download_dataset(path, user_id='delta-prox', local_dir=None, force_download=True):
    if local_dir is None:
        local_dir = os.path.join(CACHE_DIR, path)
    if os.path.exists(local_dir) and not force_download:
        return local_dir
    huggingface_hub.snapshot_download(repo_id=f"{user_id}/{path}",
                                      local_dir=local_dir,
                                      repo_type="dataset")
    return local_dir
