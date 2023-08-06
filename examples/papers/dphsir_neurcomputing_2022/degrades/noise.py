import numpy as np

class GaussianNoise:
    def __init__(self, sigma):
        np.random.seed(seed=0)  # for reproducibility
        self.sigma = sigma

    def __call__(self, img):
        img_L = img + np.random.normal(0, self.sigma, img.shape)
        return img_L