# -*- coding: UTF-8 -*-
from icv.image.io import imread
import os


class Image(object):
    def __init__(self, path=None, img=None, ext=None, name=None, category=None):
        self.path = path
        self.img = img
        self.name = name
        self.category = category
        self.ext = ext

        if self.img is None and self.path is not None:
            self.img = imread(path)

        if self.name is None and self.path is not None:
            self.name = os.path.basename(name)

        if self.ext is None and self.path is not None:
            _, self.ext = os.path.splitext(os.path.split(self.path)[1])

        self.shape = None if self.img is None else self.img.shape
        self.height = self.shape[0] if self.shape is not None else -1
        self.width = self.shape[1] if self.shape is not None else -1
        self.size = (self.height, self.width)
