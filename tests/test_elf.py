#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
test_elf.py
Authors: rensike(rensike@baidu.com)
Date:    2019/8/27 下午10:02
"""

from icv.data.elf import Elf
from icv.data.converter import ElfConverter

if __name__ == '__main__':
    from glob import glob
    import os

    image_list = glob("/Users/rensike/Work/icv/annotation/elf/segment2/*.jpg")

    image_anno_attachment_pathlist = []
    for img in image_list:
        name = os.path.basename(img).rsplit(".",1)[0]
        image_anno_attachment_pathlist.append([
            img,
            "/Users/rensike/Work/icv/annotation/elf/segment2/xml/outputs/%s.xml" % name,
            "/Users/rensike/Work/icv/annotation/elf/segment2/xml/outputs/attachments"
        ])

    elf = Elf(image_anno_attachment_pathlist)

    converter = ElfConverter(elf)
    # voc = converter.to_voc("/Users/rensike/Work/icv/elf_segment_to_voc_xml",reset=True)
    coco = converter.to_coco("/Users/rensike/Work/icv/elf_segment_to_coco_xml",reset=True)

    # voc.vis(save_dir="/Users/rensike/Work/icv/elf_segment_to_voc_xml/vis",reset_dir=True)
    coco.vis(save_dir="/Users/rensike/Work/icv/elf_segment_to_coco_xml/vis",reset_dir=True)

    # elf.vis(save_dir="/Users/rensike/Work/icv/annotation/elf/segment2_vis_xml",reset_dir=True)
