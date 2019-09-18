#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
test_draw.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/5 下午7:37
"""

from icv import imshow_bboxes

if __name__ == '__main__':
    imshow_bboxes(
        "/Users/rensike/Work/iqi/dataset/test1/203.jpg",
        bboxes=[[139,53,371,370]],
        is_show=True
    )
