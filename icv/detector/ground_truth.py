# -*- coding: UTF-8 -*-
from ..utils import is_seq
import numpy as np
from ..image.vis import imshow_bboxes


class GroundTruth(object):
    def __init__(self, bboxes, classes, image):
        assert is_seq(bboxes), "param det_bboxes should be a sequence."
        self._bboxes = np.array(bboxes)
        self._classes = np.array(classes)
        self._image = image

    @property
    def bboxes(self):
        return self._bboxes

    @property
    def classes(self):
        return self._classes

    @property
    def image(self):
        return self._image

    @property
    def image_drawed(self):
        return self._image_drawed

    def vis(self, img, is_show=False, save_path=None):
        image_drawed = imshow_bboxes(img, self.bboxes, self.classes, is_show=is_show, save_path=save_path)
        self._image_drawed = image_drawed
