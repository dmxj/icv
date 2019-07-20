# -*- coding: UTF-8 -*-
import numpy as np
from .bbox import BBox
from .polys import Polygon
from .mask import Mask

class AnnoShape(object):
    def __init__(self,bbox=None,polys=None,mask=None,label=None):
        self.bbox = None if bbox is None else BBox.init_from(bbox,label)
        self.polys = None if polys is None else Polygon.init_from(polys,label)
        self.mask = None if mask is None else Mask.init_from(mask,label)

        self.label = label

        self.seg_model_polys = self.polys is not None
        self.seg_model_mask = self.mask is not None

        assert not (self.seg_model_polys and self.seg_model_mask)

        if self.bbox is None and self.seg_model_polys:
            self.bbox = self.polys.to_bounding_box()

        if self.bbox is None and self.seg_model_mask:
            self.bbox = self.mask.to_bounding_box()








