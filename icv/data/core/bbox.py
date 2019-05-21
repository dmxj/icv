# -*- coding: UTF-8 -*-
from easydict import EasyDict as edict
import numpy as np

class BBox(object):
    def __init__(self,xmin,ymin,xmax,ymax,label=None,**kwargs):
        assert xmin <= xmax, "xmax should be large than xmin."
        assert ymin <= ymax, "ymax should be large than ymin."

        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

        self._center = ((xmax+xmin)/2,(ymax+ymin)/2)
        self._width = (xmax - xmin)/2
        self._height = (ymax - ymin)/2

        self.fields = edict()
        for k in kwargs:
            self.add_field(k,kwargs[k])

        self.lable = label
        if label:
            self.add_field("label", self.lable)

        self.add_field("xmin", self.xmin)
        self.add_field("ymin", self.ymin)
        self.add_field("xmax", self.xmax)
        self.add_field("ymax", self.ymax)
        self.add_field("center", self.center)
        self.add_field("width", self.width)
        self.add_field("height", self.height)

    @staticmethod
    def init_from(bbox):
        assert isinstance(bbox, BBox) or isinstance(bbox, list) or isinstance(bbox, np.ndarray)
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if isinstance(bbox, list):
            assert len(bbox) == 4
            bbox = BBox(xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3])
        return bbox

    def add_field(self, field, field_data):
        self.fields[field] = field_data

    def get_field(self, field):
        return self.fields[field]

    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    @property
    def bbox(self):
        return [self._xmin,self._ymin,self._xmax,self._ymax]

    @property
    def xmin(self):
        return self._xmin

    @property
    def ymin(self):
        return self._ymin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymax(self):
        return self._ymax

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def iou(self,bbox):
        bbox = BBox.init_from(bbox)
        # TODO: check it
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.xmin, bbox.xmin)
        yA = max(self.ymin, bbox.ymin)
        xB = min(self.xmax, bbox.xmax)
        yB = min(self.ymax, bbox.ymax)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)
        boxBArea = (bbox.xmax - bbox.xmin + 1) * (bbox.ymax - bbox.ymin + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "xmin={}, ".format(self.xmin)
        s += "ymin={}, ".format(self.ymin)
        s += "xmax={}, ".format(self.xmax)
        s += "ymax={}, ".format(self.ymax)
        return s


