# -*- coding: UTF-8 -*-
import numpy as np

from .colorspace import bgr2rgb, rgb2bgr
from ..io import imread

def imnormalize(img, mean, std, to_rgb=True):
    img = imread(img).astype(np.float32)
    if to_rgb:
        img = bgr2rgb(img)
    return (img - mean) / std


def imdenormalize(img, mean, std, to_bgr=True):
    img = (imread(img) * std) + mean
    if to_bgr:
        img = rgb2bgr(img)
    return img
