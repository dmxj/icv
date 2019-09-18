#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
target.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/14 下午5:42
"""


class Target(object):
    area = 0

    @property
    def is_small(self):
        return hasattr(self, "area") and self.area < 322

    @property
    def is_middle(self):
        return hasattr(self, "area") and 322 <= self.area < 962

    @property
    def is_large(self):
        return hasattr(self, "area") and self.area >= 962
