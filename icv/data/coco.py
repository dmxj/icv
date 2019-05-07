# -*- coding: UTF-8 -*-
import os
import json
from .dataset import IcvDataSet
from .core.sample import Sample
from .core.bbox import BBox
from .core.bbox_list import BBoxList
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from icv.image import imread,imshow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

class Coco(IcvDataSet):
    def __init__(self,image_dir,anno_file,keep_no_anno_image=True,one_index=False,transform=None,target_transform=None):
        assert os.path.isdir(image_dir), "param image_dir is not a dir!"
        assert os.path.exists(anno_file), "param anno_file is not exist!"

        self.image_dir = image_dir
        self.anno_file = anno_file

        self.keep_no_anno_image = keep_no_anno_image
        self.coco = COCO(self.anno_file)
        self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = target_transform

        self.categories = self.get_categories()
        self.ids = self.coco.getImgIds()
        super(Coco, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index)

    def get_categories(self):
        categories = []
        for catid in self.coco.getCatIds():
            categories.append(self.coco.cats[catid]["name"])
        return categories

    def get_sample(self,id):
        path = self.coco.loadImgs(id)[0]['file_name']
        img = imread(os.path.join(self.image_dir, path))

        anns = self.coco.imgToAnns[id]
        bbox_list = []
        for ann in anns:
            xmin,ymin,width,height = ann["bbox"]
            cat = self.id2cat[ann["category_id"]]
            bbox_list.append(BBox(xmin,ymin,xmin+width,ymin+height,label=cat,**ann))

        img_info = self.coco.imgs[id]
        sample = Sample.init(path.rsplit(".",1)[0],BBoxList(bbox_list=bbox_list),img,**img_info)
        sample.id = id
        sample.add_field("anns",anns)
        return sample

    def _write_sample(self, anno_samples, dist_path):
        annotation = {
            "images":[],
            "annotations":[],
            "categories":[],
        }

        anno_id = 0
        cats = {}
        for id,sample in enumerate(anno_samples):
            if sample.has_field("file_name"):
                continue
            img_height,img_width = sample.image.shape[:2]
            annotation["images"].append(
                {
                    "id":id,
                    "file_name":sample.get_field("file_name"),
                    "width":img_width,
                    "height":img_height
                }
            )

            for bbox in sample.bbox_list:
                if bbox.has_field("label"):
                    continue
                anno = {
                    "bbox":[bbox.xmin,bbox.ymin,bbox.width,bbox.height]
                }

                if bbox.has_field("segmentation"):
                    anno["segmentation"] = bbox.get_field("segmentation")

                if bbox.has_field("area"):
                    anno["area"] = bbox.get_field("area")

                if bbox.has_field("iscrowd"):
                    anno["iscrowd"] = bbox.get_field("iscrowd")

                anno["category_id"] = self.cat2id[bbox.lable]
                anno["image_id"] = id
                anno["id"] = anno_id
                anno_id += 1

                if bbox.lable not in cats:
                    cats[bbox.lable] = {"supercategory": "", "id": anno["category_id"], "name": bbox.lable}

        annotation["categories"] = cats.values()
        json.dump(annotation,open(dist_path,"w"))

    def showAnns(self, id):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        sample = self.get_sample(id)
        imshow(sample.image)
        self.coco.showAnns()


    def vis(self,save_dir=None):
        pass