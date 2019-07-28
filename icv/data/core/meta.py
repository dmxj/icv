# -*- coding: UTF-8 -*-
from abc import ABCMeta, abstractclassmethod
from icv.utils import EasyDict


class Meta(object):
    __metaclass__ = ABCMeta


class SampleMeta(EasyDict):
    def dict(self):
        return super(SampleMeta, self).dict()


class AnnoMeta(EasyDict):
    def dict(self):
        return super(AnnoMeta, self).dict()
