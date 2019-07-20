# -*- coding: UTF-8 -*-
from abc import ABCMeta,abstractclassmethod
from icv.utils import EasyDict

class Meta(object):
    __metaclass__ = ABCMeta

class SampleMeta(EasyDict):
    pass

class AnnoMeta(EasyDict):
    pass



