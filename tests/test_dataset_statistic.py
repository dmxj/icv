#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
test_dataset_statistic.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/14 下午6:46
"""

from icv.data import Coco,Voc

if __name__ == '__main__':
    # voc = Voc("/Users/rensike/Work/宝钢热轧/baogang_hot_data_v4/pascal_voc_hot_v4_0")
    # voc.statistic(is_plot_show=True,plot_save_path="./hot_voc_statics_plt.png")

    voc = Voc("/Users/rensike/Work/icv/elf_segment_to_voc_json",mode="seg")
    voc.statistic(is_plot_show=True,plot_save_path="./voc_statics_plt.png")

    # coco = Coco(
    #     image_dir="/Users/rensike/Work/icv/elf_segment_to_coco_xml/images/trainval",
    #     anno_file="/Users/rensike/Work/icv/elf_segment_to_coco_xml/annotations/trainval.json"
    # )
    # coco.statistic(is_plot_show=True,plot_save_path="./coco_statics_plt.png")
