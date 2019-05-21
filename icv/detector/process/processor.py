# -*- coding: utf-8 -* -
"""
模型输入前处理，以及模型预测后处理
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from abc import *


class Processor(abc.ABCMeta):

    @abstractmethod
    def pre_process(self, inputs):
        pass

    @abstractmethod
    def post_process(self, outputs):
        pass
