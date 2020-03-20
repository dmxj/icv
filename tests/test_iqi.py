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
from icv.data.core import BBoxList,BBox
from icv.image import imread_tob64, imdraw_polygons_with_bbox,imshow_bboxes
import json

def req_od(image_file,srv_addr):
    # res = do_post(srv_addr, data={
    #     "image": imread_tob64(image_file),
    # }, json=True)

    res = do_post(srv_addr, data={
        "inputs": [{'data':imread_tob64(image_file)}],
    }, json=True)

    if res.status_code != 200:
        print("request error:", res.error)
        pass

    res_data = json.loads(res.content)
    # data = res_data["data"]
    data = res_data["predictResult"][0]
    print("result data:", res_data)
    print("result data:", data)

    bboxes = BBoxList([BBox(_[1], _[0], _[3], _[2]) for _ in data["detection_boxes"]])

    imshow_bboxes(
        image_file,
        bboxes=bboxes,
        scores=data["detection_scores"],
        classes=data["detection_classes"],
        is_show=True,
        save_path="./od_infer_result.jpg"
    )

def req_seg(image_file,srv_addr):
    res = do_post(srv_addr, data={
        "image": imread_tob64(image_file),
    }, json=True)

    if res.status_code != 200:
        print("request error:", res.error)
        pass

    res_data = json.loads(res.content)
    segmap = res_data["data"]
    print("result data:", res_data)

    polygons = []
    name_list = []

    for c in segmap:
        for polygon in segmap[c]:
            name_list.append(c)
            polygons.append(polygon)

    imdraw_polygons_with_bbox(
        image_file,
        polygons,
        label_list=name_list,
        is_show=True,
        save_path="./seg_infer_result.jpg"
    )

if __name__ == '__main__':
    # test_image = "/Users/rensike/Work/iqi/seginfer/test_seg.jpg"
    # endpoint = "http://127.0.0.1:9527/api/v1/predict"

    req_od(
        image_file="/Users/rensike/Files/data/images/018.jpg",
        srv_addr="http://szth-bcc-online-com0-1322.szth:8012/v1/models/h7ja5kvyf93ilpx6ozgrw2esmncu0bt8/predict",
        # srv_addr = "http://127.0.0.1:9527/api/v1/predict"
    )

    # req_seg(
    #     image_file="/Users/rensike/Files/data/images/018.jpg",
    #     # image_file="/Users/rensike/Work/iqi/big.jpg",
    #     srv_addr="http://172.24.155.76:9527/api/MTcyLjE5LjAuNDo5NTI3/api/v1/predict",
    # )

