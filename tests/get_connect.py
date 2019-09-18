#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
get_connect.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/12 下午6:59
"""
from skimage import measure
import cv2
import numpy as np
from icv.image.vis import imshow

def counters(img_pth):
    img = cv2.imread(img_pth, 0)

    _, labels = cv2.connectedComponents(img)
    ids = list(set(labels.flatten().tolist()))
    print("ids: ", ids, ", labels: ", labels)

    for id in ids[1:]:
        w = np.where(labels == id)
        ground_truth_binary_mask = np.zeros_like(img, np.uint8)
        ground_truth_binary_mask[w] = 1
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)
        print(len(contours))

if __name__ == '__main__':
    # imshow("/Users/rensike/Work/radar/1.jpg")

    img = cv2.imread("/Users/rensike/Work/radar/0.png", 0)

    # w = np.where(img != 0)
    #
    # ymin = min(w[0])
    # ymax = max(w[0])
    #
    # xmin = min(w[1])
    # xmax = max(w[1])
    #
    # img1 = img[ymin:ymax,xmin:xmax]

    _, labels = cv2.connectedComponents(img)
    ids = list(set(labels.flatten().tolist()))

    max_id = 0
    max_cnt = -1
    for id in ids[1:]:
        w = np.where(labels == id)
        if img[w[0][0],w[1][0]] != 0:
            if len(w[0]) > max_cnt:
                max_cnt = len(w[0])
                max_id = id

    w = np.where(labels == max_id)
    ymin = min(w[0])
    ymax = max(w[0])

    xmin = min(w[1])
    xmax = max(w[1])

    img1 = img[ymin:ymax,xmin:xmax]

    imshow(img1)

