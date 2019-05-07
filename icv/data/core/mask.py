# -*- coding: UTF-8 -*-
import torch
import numpy as np
import cv2
import pycocotools.mask as mask_utils

class Mask(object):
    """
    Binary Mask
    """
    def __init__(self,mask):
        self.mask = np.array(mask)
        assert len(self.mask.shape) == 2
        self.size = tuple(self.mask.shape)

    def find_contours(self):
        mask = cv2.UMat(self.mask)
        contour, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )

        reshaped_contour = []
        for entity in contour:
            assert len(entity.shape) == 3
            assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
            reshaped_contour.append(entity.reshape(-1).tolist())
        return reshaped_contour



