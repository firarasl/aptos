import cv2
import numpy as np
import torchvision.transforms as T

from aptos.utils import setup_logger


class ImgProcessor:
    """
    This class is responsible for preprocessing the images, eg. crop, sharpen, resize, normalise.
    """

    def __init__(self, crop_tol=12, img_width=600, verbose=0):
        self.logger = setup_logger(self, verbose)
        self.crop_tol = crop_tol
        self.img_width = img_width
        self.sequential = T.Compose([
            self.read_png,
            self.crop_black,
            self.crop_square,
            self.resize
        ])

    def __call__(self, filename):
        return self.sequential(filename)

    def read_png(self, filename):
        """
        Load the image into a numpy array, and switch the channel order so it's in the format
        expected by matplotlib (rgb).
        """
        return cv2.imread(filename)[:, :, ::-1]  # bgr => rgb

    def crop_black(self, img):
        """
        Apply a bounding box to crop empty space around the image. In order to find the bounding
        box, we blur the image and then apply a threshold. The blurring helps avoid the case where
        an outlier bright pixel causes the bounding box to be larger than it needs to be.
        """
        gb = cv2.GaussianBlur(img, (7, 7), 0)
        mask = (gb > self.crop_tol).any(2)
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return img[y0:y1, x0:x1]


    def crop_square(self, img):
        """
        Crop the image to a square (cutting off sides of a circular image).
        """
        H, W, C = img.shape
        crop_size = min(int(W * 0.87), H)
        if W <= crop_size:
            x0 = 0
            x1 = W
        else:
            width_excess = W - crop_size
            x0 = width_excess // 2
            x1 = min(x0 + crop_size, W)
        if H <= crop_size:
            y0 = 0
            y1 = H
        else:
            height_excess = H - crop_size
            y0 = height_excess // 2
            y1 = min(y0 + crop_size, H)
        return img[y0:y1, x0:x1]

    def resize(self, img):
        dim = (self.img_width, self.img_width)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

