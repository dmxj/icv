#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Setup script.
Authors: rensike(rensike@baidu.com)
Date:    2019/7/19 下午11:31
"""

from icv.utils import do_post
from icv.image import imread_tob64, imdraw_polygons_with_bbox
import json

if __name__ == '__main__':
    test_image = "/Users/rensike/Work/iqi/seginfer/test_seg.jpg"
    endpoint = "http://127.0.0.1:9527/api/v1/predict"

    res = do_post(endpoint, data=dict(image=imread_tob64(test_image)), json=True)
    res_data = json.loads(res.content)

    segmap = res_data["data"]

    polygons = []
    name_list = []

    for c in segmap:
        for polygon in segmap[c]:
            name_list.append(c)
            polygons.append(polygon)

    imdraw_polygons_with_bbox(
        test_image,
        polygons,
        name_list=name_list,
        is_show=True,
    )
