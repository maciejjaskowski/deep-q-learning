import numpy as np
import skimage.transform

class Phi4(object):

    def __init__(self, method):
        self.screen_size = (84, 84)
        self.method = method

    def __call__(self, frame):
        return self.resize_and_crop(frame).reshape(*((1, 1) + self.screen_size))

    def resize_and_crop(self, im):
        # Resize so smallest dim = 256, preserving aspect ratio
        if self.method == "resize":
            return (skimage.transform.resize(im, (84, 84)) * 255).astype(dtype=np.uint8)
        else:
            im = im[40:-10, :]
            h, w = im.shape
            if h < w:
                im = skimage.transform.resize(im, (84, w*84//h), preserve_range=True)
            else:
                im = skimage.transform.resize(im, (h*84//w, 84), preserve_range=True)

            # Central crop to 224x224
            h, w = im.shape
            return im[h//2-42:h//2+42, w//2-42:w//2+42].astype(dtype=np.uint8)
