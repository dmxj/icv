# -*- coding: UTF-8 -*-
import numpy as np
from icv.data.core.polygon import PolygonInstance
from icv.data.core import FLIP_LEFT_RIGHT,FLIP_TOP_BOTTOM
from icv.data.core.mask_list import MaskList

class PolygonList(object):
    """
    This class handles PolygonInstances for all objects in the image
    """

    def __init__(self, polygons, size):
        """
        Arguments:
            polygons:
                a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.
                OR
                a list of PolygonInstances.
                OR
                a PolygonList
            size: absolute image size
        """
        if isinstance(polygons, (list, tuple)):
            if len(polygons) == 0:
                polygons = [[[]]]
            if isinstance(polygons[0], (list, tuple)):
                assert isinstance(polygons[0][0], (list, tuple)), str(
                    type(polygons[0][0])
                )
            else:
                assert isinstance(polygons[0], PolygonInstance), str(type(polygons[0]))

        elif isinstance(polygons, PolygonList):
            size = polygons.size
            polygons = polygons.polygons

        else:
            RuntimeError(
                "Type of argument `polygons` is not allowed:%s" % (type(polygons))
            )

        assert isinstance(size, (list, tuple)), str(type(size))

        self.polygons = []
        for p in polygons:
            p = PolygonInstance(p, size)
            if len(p) > 0:
                self.polygons.append(p)

        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        for polygon in self.polygons:
            flipped_polygons.append(polygon.transpose(method))

        return PolygonList(flipped_polygons, size=self.size)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_polygons = []
        for polygon in self.polygons:
            cropped_polygons.append(polygon.crop(box))

        cropped_size = w, h
        return PolygonList(cropped_polygons, cropped_size)

    def resize(self, size):
        resized_polygons = []
        for polygon in self.polygons:
            resized_polygons.append(polygon.resize(size))

        resized_size = size
        return PolygonList(resized_polygons, resized_size)

    def to(self, *args, **kwargs):
        return self

    def convert_to_binarymask(self):
        if len(self) > 0:
            masks = np.stack([p.convert_to_binarymask() for p in self.polygons])
        else:
            size = self.size
            masks = np.empty([0, size[1], size[0]], dtype=np.uint8)

        return MaskList(masks)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, item):
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item,np.uint8):
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return PolygonList(selected_polygons, size=self.size)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s