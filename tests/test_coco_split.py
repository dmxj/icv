#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
test_coco_split.py
Authors: rensike(rensike@baidu.com)
Date:    2019/10/15 下午5:36
"""

from icv.data.coco import Coco
from icv.data.voc import Voc
import random

def random_sample(seq,ratio):
    count = min(len(seq),int(len(seq)*ratio))
    return random.sample(seq,count)

if __name__ == '__main__':
    # coco = Coco(
    #     image_dir="/Users/rensike/Work/icv/labelme_seg_to_coco/images/train",
    #     anno_file="/Users/rensike/Work/icv/labelme_seg_to_coco/annotations/train.json"
    # )
    # coco.divide("/Users/rensike/Work/icv/labelme_seg_to_coco_split",splits=["train","test","val"],ratios=[0.8,0.1,0.1])

    voc = Voc("/Users/rensike/Work/icv/labelme_seg_to_voc",split="train",mode="segmentation")
    voc.divide("/Users/rensike/Work/icv/labelme_seg_to_voc_split",splits=["train","test","val"],ratios=[0.8,0.1,0.1])

    # train_coco = coco.copy()
    # test_coco = coco.copy()
    # val_coco = coco.copy()
    #
    # ids = coco.ids
    # train_ids = random_sample(ids,0.8)
    # test_ids = random_sample([_ for _ in ids if _ not in train_ids],0.5)
    # val_ids = random_sample([_ for _ in ids if _ not in train_ids and _ not in test_ids],0.5)
    #
    # print(list(coco.sample_db.keys()))
    #
    # print("train_ids:",train_ids)
    # print("test_ids:",test_ids)
    # print("val_ids:",val_ids)
    #
    # train_coco.keep(train_ids)
    # test_coco.keep(test_ids)
    # val_coco.keep(val_ids)
    #
    # print("train_coco.ids:",train_coco.ids)
    # print("test_coco.ids:",test_coco.ids)
    # print("val_coco.ids:",val_coco.ids)
    #
    # train_coco.save("/Users/rensike/Work/icv/labelme_seg_to_coco_split",reset_dir=True,split="train")
    # test_coco.save("/Users/rensike/Work/icv/labelme_seg_to_coco_split",split="test")
    # val_coco.save("/Users/rensike/Work/icv/labelme_seg_to_coco_split",split="val")


