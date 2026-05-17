import numpy as np
import cv2
import yaml
import albumentations as A
from scipy.interpolate import UnivariateSpline


class PhotometricSimulator:
    def __init__(self, str_crf_filepath):
        self.str_crf_filepath = str_crf_filepath
        self.g_lookup = self.generate_crf_lookup()
        self.img_lb = 0
        self.img_ub = 65535
        self.irr_lb = self.g_lookup[self.img_lb]
        self.irr_ub = self.g_lookup[self.img_ub]

    def img2irr(self, img: np.ndarray):
        return self.g_lookup[img]

    def irr2img(self, irr: np.ndarray):
        indices = np.searchsorted(self.g_lookup, irr, side="right") - 1
        img_n = indices / 65535.0
        return img_n

    def img_synthesis(self, img0: np.ndarray, expo0, expo1):
        irr0 = self.img2irr(img0)
        irr1 = irr0 - np.log(expo0) + np.log(expo1)
        irr1 = np.clip(irr1, self.irr_lb, self.irr_ub)
        img1_n_syn = self.irr2img(irr1)
        return img1_n_syn

    def generate_crf_lookup(self):
        with open(self.str_crf_filepath, "r") as f:
            data = yaml.safe_load(f.read())
            g_func_y = np.array(data["g_func"], np.float64)

        g_func_x = np.arange(0, 256, 1) / 255.0

        g_func = UnivariateSpline(
            g_func_x[:-1],
            g_func_y[:-1],
            s=0.001,
            k=5,
        )

        int_x_n = np.arange(0, 65536, 1).astype(np.float64) / 65535.0
        g_lookup = g_func(int_x_n)
        return g_lookup


class Exposition(A.ImageOnlyTransform):
    def __init__(self, crf_path="crf.yaml", expo0=1.0, p=0.3):
        super().__init__(p=p)
        self.ps = PhotometricSimulator(crf_path)
        self.expo0 = expo0

    def apply(self, img, **params):
        img_u8 = img.astype(np.uint8)
        expo_shift = np.random.uniform(-0.5, 0.5)
        expo1 = 2 ** expo_shift
        img_u16 = img_u8.astype(np.uint16) * 257
        img_aug_n = self.ps.img_synthesis(img_u16, self.expo0, expo1)
        img_aug = np.clip(img_aug_n * 255.0, 0, 255).astype(np.uint8)

        return img_aug