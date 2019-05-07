# -*- coding: UTF-8 -*-
import numpy as np
import copy
import pycocotools.mask as mask_utils
from .mask import Mask
from . import FLIP_TOP_BOTTOM,FLIP_LEFT_RIGHT

class PolygonInstance(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size):
        """
            Arguments:
                a list of lists of numbers.
                The first level refers to all the polygons that compose the
                object, and the second level to the polygon coordinates.
        """
        if isinstance(polygons, (list, tuple)):
            valid_polygons = []
            for p in polygons:
                # p = torch.as_tensor(p, dtype=torch.float32)
                if len(p) >= 6:  # 3 * 2 coordinates
                    valid_polygons.append(p)
            polygons = valid_polygons

        elif isinstance(polygons, PolygonInstance):
            polygons = copy.copy(polygons.polygons)

        else:
            RuntimeError(
                "Type of argument `polygons` is not allowed:%s" % (type(polygons))
            )

        """ This crashes the training way too many times...
        for p in polygons:
            assert p[::2].min() >= 0
            assert p[::2].max() < size[0]
            assert p[1::2].min() >= 0
            assert p[1::2].max() , size[1]
        """

        self.polygons = polygons
        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return PolygonInstance(flipped_polygons, size=self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, np.ndarray)), str(type(box))
        box = list(box)

        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = map(float, box)

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        w, h = xmax - xmin, ymax - ymin

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - xmin  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - ymin  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return PolygonInstance(cropped_polygons, size=(w, h))

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return PolygonInstance(scaled_polys, size)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return PolygonInstance(scaled_polygons, size=size)

    def convert_to_binarymask(self):
        width, height = self.size
        # formatting for COCO PythonAPI
        polygons = [p.numpy() for p in self.polygons]
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return Mask(mask)

    def __len__(self):
        return len(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_groups={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s



