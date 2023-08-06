import numpy as np


class RandomMask:
    def __init__(self, ratio=0.2):
        """ ratio: ratio to keep
        """
        self.ratio = ratio

    def __call__(self, img):
        mask = (np.random.rand(*img.shape) > (1-self.ratio)).astype('float')
        img_L = img * mask
        return img_L, mask


class StripeMask:
    """ Input: [W,H,B] """

    def __init__(self, bandwise=False):
        self.bandwise = bandwise

    def __call__(self, img):
        mask = np.ones_like(img)
        for i in range(0, img.shape[1]-5, 10):
            mask[:, i+3] = 0
            mask[:, i+4] = 0
            mask[:, i+5] = 0
        img_L = img * mask
        return img_L, mask


class RandomStripe:
    """ Input: [W,H,B] """

    def __init__(self, num_bands=4, bandwise=True, ratio=0.2):
        self.num_bands = num_bands  # how many bands will be added stripe noise
        self.bandwise = bandwise  # if the location of stripe noise are the same
        self.ratio = ratio

    def __call__(self, img):
        mask = np.ones_like(img)
        w, h, b = img.shape
        # random select 4 band to add stripe (actually dead line)
        # start_band = np.random.choice(b-self.num_bands, replace=False)
        for i in range(b):
            stripes = np.random.choice(h, int(h*self.ratio), replace=False)
            for j in stripes:
                mask[:, j:j+4, i] = 0
        img_L = img * mask
        return img_L, mask


class FastHyStripe:
    """ Input: [W,H,B] """

    def __init__(self, num_bands=15, bandwise=False):
        self.num_bands = num_bands  # how many bands will be added stripe noise
        self.bandwise = bandwise  # if the location of stripe noise are the same

    def __call__(self, img):
        import time
        np.random.seed(int(time.time()))
        mask = np.ones_like(img)
        w, h, b = img.shape
        # random select 4 band to add stripe (actually dead line)

        start_band = 10
        # start_band = 0
        for i in range(start_band, start_band+self.num_bands):
            stripes = np.random.choice(h, 20, replace=False)
            for k, j in enumerate(stripes):
                t = np.random.rand()
                if k == 4:
                    mask[:, j:j+30, i] = 0
                elif k == 10:
                    mask[:, j:j+15, i] = 0
                elif t > 0.6:
                    mask[:, j:j+4, i] = 0
                else:
                    mask[:, j:j+2, i] = 0
            if self.bandwise:
                break
        if self.bandwise:
            mask[:, :, start_band:start_band+self.num_bands] = np.expand_dims(mask[:, :, start_band], axis=-1)
        img_L = img * mask
        return img_L, mask
