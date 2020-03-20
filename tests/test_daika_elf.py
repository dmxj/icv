#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
test_daika_elf.py
Authors: rensike(rensike@baidu.com)
Date:    2019/10/15 下午3:06
"""

from icv.data.elf import Elf
from icv.data.converter import ElfConverter
import os
from glob import glob
from icv.image.vis import imshow_bboxes
import numpy as np

if __name__ == '__main__':
    data_path = "/Users/rensike/Work/daika/test/"

    # imshow_bboxes(data_path + "images/戴卡_18.jpg", bboxes=np.array([[1983, 1207, 1984, 1209]]),
    #               save_path="./dakai18.jpg")

    image_list = glob(data_path + "images/*.jpg")

    image_anno_attachment_pathlist = []
    for img in image_list:
        name = os.path.basename(img).rsplit(".", 1)[0]
        image_anno_attachment_pathlist.append([
            img,
            data_path + "outputs/%s.json" % name,
            data_path + "outputs/attachments"
        ])

    elf = Elf(image_anno_attachment_pathlist)
    # print(elf.categories)

    elf.vis(save_dir="./daikai/")

    # converter = ElfConverter(elf)
    # # voc = converter.to_voc("/Users/rensike/Work/icv/elf_segment_to_voc_xml",reset=True)
    # coco = converter.to_coco(data_path + "to_coco", reset=True)
    # coco.vis(save_dir=data_path + "to_coco_vis")
