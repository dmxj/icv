# -*- coding: UTF-8 -*-
import torch
import numpy as np
import cv2
import pycocotools.mask as mask_utils
from .mask import Mask
from . import FLIP_TOP_BOTTOM,FLIP_LEFT_RIGHT

class MaskList(object):
    """
    Binary Mask List
    """
    def __init__(self,masks):
        if isinstance(masks,MaskList):
            self.masks = masks.masks
        elif isinstance(masks, (list, tuple)):
            if isinstance(masks[0],np.ndarray):
                assert all([m.shape == masks[0].shape for m in masks])
                masks = np.array(masks)
                if len(masks.shape) == 2:
                    self.masks = masks[None]
            elif isinstance(masks[0], dict) and "counts" in masks[0]:
                masks = mask_utils.decode(masks)  # [h, w, n]
                self.masks = masks[...,[2,0,1]]

            assert len(self.masks.shape) == 3
            self.count = self.masks.shape[0]
            self.size = (self.masks.shape[1],self.masks.shape[2])

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = torch.Tensor(self.masks).flip(dim).detach().numpy()
        return MaskList(flipped_masks)

    def crop(self, box):
        assert isinstance(box, (list, tuple)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_masks = self.masks[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return MaskList(cropped_masks)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        # TODO: 双线性插值的包存在
        resized_masks = torch.nn.functional.interpolate(
            input=self.masks[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].type_as(self.masks)
        return MaskList(resized_masks)

    def _findContours(self):
        contours = []
        masks = self.masks.detach().numpy()
        for mask in masks:
            mask = cv2.UMat(mask)
            contour, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )

            reshaped_contour = []
            for entity in contour:
                assert len(entity.shape) == 3
                assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
                reshaped_contour.append(entity.reshape(-1).tolist())
            contours.append(reshaped_contour)
        return contours

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        # Probably it can cause some overhead
        # but preserves consistency
        masks = self.masks[index].clone()
        return MaskList(masks)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[1])
        s += "image_height={})".format(self.size[0])
        return s


