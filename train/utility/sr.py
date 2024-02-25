import cv2
import numpy as np
import scipy.ndimage

# C,H,W format


"""For Super-Resolution"""


class SRDegrade(object):
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def __call__(self, img):
        from scipy.ndimage import zoom
        img = zoom(img, zoom=(1, 1. / self.scale_factor, 1. / self.scale_factor))
        img = zoom(img, zoom=(1, self.scale_factor, self.scale_factor))
        return img


class GaussianBlurScipy(object):
    def __init__(self, ksize=8, sigma=3):
        self.sigma = sigma
        self.truncate = (((ksize - 1) / 2) - 0.5) / sigma

    def __call__(self, img):
        from scipy.ndimage.filters import gaussian_filter
        img = gaussian_filter(img, sigma=self.sigma, truncate=self.truncate)
        return img


# ---------------------- Blur ----------------------- #

def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1),
                         np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


class AbstractBlur:
    def __call__(self, img):
        img_L = scipy.ndimage.filters.convolve(
            img, np.expand_dims(self.kernel, axis=0), mode='wrap')
        return img_L


class GaussianBlur(AbstractBlur):
    def __init__(self, ksize=8, sigma=3):
        self.kernel = fspecial_gaussian(ksize, sigma)


class UniformBlur(AbstractBlur):
    def __init__(self, ksize):
        self.kernel = np.ones((ksize, ksize)) / (ksize * ksize)


## -------------------- Resize -------------------- ##

class KFoldDownsample:
    ''' k-fold downsampler:
        Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others 
    '''

    def __init__(self, sf):
        self.sf = sf

    def __call__(self, img):
        st = 0
        return img[:, st::self.sf, st::self.sf]


class Resize:
    def __init__(self, sf, mode='cubic'):
        self.sf = sf
        self.mode = self.mode_map[mode]

    def __call__(self, img):
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (int(img.shape[1] * self.sf), int(img.shape[0] * self.sf)), interpolation=self.mode)
        img = img.transpose(2, 0, 1)
        return img

    mode_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }


class BicubicDownsample(Resize):
    def __init__(self, sf):
        super().__init__(1 / sf, 'cubic')

    def __call__(self, img):
        return super().__call__(img)


class BicubicUpsample(Resize):
    def __init__(self, sf):
        super().__init__(sf, 'cubic')

    def __call__(self, img):
        return super().__call__(img)
