# -*- coding: UTF-8 -*-
from icv.data import Coco, Voc, LabelMe, VocConverter, CocoConverter,LabelMeConverter
import os
from glob import glob

def voc_to_coco():
    voc_path = "/Users/rensike/Files/temp/voc_tiny"
    voc1 = Voc(voc_path, mode="segment")

    # voc1.vis(save_dir="/Users/rensike/Work/icv/voc_tiny_vis")

    voc1_sub = voc1.sub(10)
    print("voc1_sub.ids:", voc1_sub.ids)

    voc1_sub.vis(save_dir="/Users/rensike/Work/icv/voc_tiny_sub1_vis", reset_dir=True)

    conveter = VocConverter(voc1_sub)

    coco = conveter.to_coco("/Users/rensike/Work/icv/voc_to_coco_test", split="val")

    coco.statistic(print_log=True)

    coco.vis(save_dir="/Users/rensike/Work/icv/voc_to_coco_test_vis", reset_dir=True)


def coco_to_voc():
    coco_image_dir = "/Users/rensike/Files/temp/coco_tiny/test"
    coco_anno_file = "/Users/rensike/Files/temp/coco_tiny/annotations/instances_test.json"

    coco = Coco(coco_image_dir, coco_anno_file, split="test")

    coco.vis(save_dir="/Users/rensike/Work/icv/coco_tiny_vis22", reset_dir=True)

    voc = CocoConverter(coco).to_voc("/Users/rensike/Work/icv/coco_to_voc_test22", split="train")

    # voc.statistic(print_log=True)
    #
    # voc.vis(save_dir="/Users/rensike/Work/icv/coco_to_voc_vis", reset_dir=True)

def labelme_to_coco_voc():
    root_dir = "/Users/rensike/Work/icv/annotation/labelme/segment"
    image_list = glob(root_dir + "/*.jpg")
    anno_list = [root_dir + "/%s.json" % os.path.basename(_).rsplit(".", 1)[0] for _ in image_list]

    image_anno_path_list = list(zip(image_list, anno_list))

    labelme = LabelMe(image_anno_path_list)

    # labelme.vis(save_dir="/Users/rensike/Work/icv/labelme_seg_vis",reset_dir=True)

    converter = LabelMeConverter(labelme)

    # coco = converter.to_coco("/Users/rensike/Work/icv/labelme_seg_to_coco",split="train",reset=True)
    # coco.vis(save_dir="/Users/rensike/Work/icv/labelme_seg_to_coco_vis",reset_dir=True)

    voc = converter.to_voc("/Users/rensike/Work/icv/labelme_seg_to_voc",split="train",reset=True)
    voc.vis(save_dir="/Users/rensike/Work/icv/labelme_seg_to_voc_vis",reset_dir=True)

if __name__ == '__main__':
    # voc_to_coco()
    # coco_to_voc()

    labelme_to_coco_voc()
