# -*- coding: UTF-8 -*-

import os
from icv.image import imread,imshow,imwrite,imshow_bboxes
from .bbox_list import BBoxList
from easydict import EasyDict as edict

class Sample(object):
    def __init__(self,name,bbox_list,image):
        if not isinstance(bbox_list,BBoxList):
            bbox_list = BBoxList(bbox_list)
        self.name = name
        self.bbox_list = bbox_list
        self.image = imread(image)

        self.fields = edict()
        self.add_field("name",self.name)
        self.add_field("bbox_list",self.bbox_list)
        self.add_field("image",self.image)

    @staticmethod
    def init(name,bbox_list,image,**kwargs):
        sample = Sample(name,bbox_list,image)
        for key in kwargs:
            sample.add_field(key,kwargs[key])
        return sample

    @property
    def shape(self):
        return self.image.shape

    @property
    def count(self):
        return self.bbox_list.length

    def __getitem__(self, item):
        if self.has_field(item):
            return self.get_field(item)
        return None

    def __setitem__(self, key, value):
        self.add_field(key, value)

    def add_field(self, field, field_data):
        self.fields[field] = field_data

    def get_field(self, field):
        return self.fields[field]

    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    def vis(self,is_show=False,save_path=None):
        image_drawed = imshow_bboxes(self.image,self.bbox_list.bbox_list,is_show=is_show,save_path=save_path)
        return image_drawed

