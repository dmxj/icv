# -*- coding: UTF-8 -*-
from .dataset import IcvDataSet
import os
import shutil
import cv2
import numpy as np
from icv.utils import mask as mask_util
from icv.utils import load_voc_anno,make_empty_voc_anno
from icv.image import imread,imshow_bboxes
from .core.sample import Sample
from .core.bbox import BBox
from .core.bbox_list import BBoxList
from lxml.etree import Element,SubElement,ElementTree
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# TODO: 分割数据（图像目录、segment=1）
class Voc(IcvDataSet):
    def __init__(self,data_dir,split=None,use_difficult=True,keep_no_anno_image=True,mode="detect",include_segment=True,categories=None,one_index=False,image_suffix="jpg"):
        self.root = data_dir
        self.split = split if split else "trainval"
        self.mode = mode
        self.include_segment = include_segment
        self.image_set = "Main" if self.mode == "detect" else "Segmentation"

        self.use_difficult = use_difficult
        self.keep_no_anno_image = keep_no_anno_image

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s." + image_suffix)
        self._seg_class_imgpath = os.path.join(self.root, "SegmentationClass", "%s.png")
        self._seg_object_imgpath = os.path.join(self.root, "SegmentationObject", "%s.png")
        self._imgsetpath = os.path.join(self.root, "ImageSets", self.image_set, "%s.txt")

        with open(self._imgsetpath % self.split) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.samples = self.get_samples()
        self.categories = categories if categories else self.get_categories()
        super(Voc, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index)

    def get_categories(self):
        categories = []
        for anno_sample in self.samples:
            label_list = anno_sample.bbox_list.labels
            if label_list:
                categories.extend(label_list)
        categories = list(set(categories))
        categories.sort()
        return categories

    def get_sample(self,id):
        anno_file = self._annopath % id
        image_file = self._imgpath % id

        if not os.path.exists(image_file):
            raise FileNotFoundError("image file : {} not exist!".format(image_file))

        image_np = imread(image_file)
        size = image_np.shape
        image_height,image_width = size[:2]
        image_depth = 1
        if len(size) == 3:
            image_depth = size[-1]

        if not os.path.exists(anno_file):
            anno_data = make_empty_voc_anno()
        else:
            anno_data = load_voc_anno(anno_file)["annotation"]

        anno_data["size"]["width"] = image_width
        anno_data["size"]["height"] = image_height
        anno_data["size"]["depth"] = image_depth

        anno_sample = Sample.init(
            name=os.path.basename(image_file).rsplit(".",1)[0],
            bbox_list=BBoxList(
                bbox_list=[
                    BBox(
                        xmin=int(object["bndbox"]["xmin"]),
                        ymin=int(object["bndbox"]["ymin"]),
                        xmax=int(object["bndbox"]["xmax"]),
                        ymax=int(object["bndbox"]["ymax"]),
                        label=object["name"],
                        **anno_data
                    )
                    for object in anno_data["object"]
                    if "difficult" not in object or object["difficult"] == '0' or (object["difficult"] != '0' and self.use_difficult)
                ]
            ),
            image=image_file,
            **anno_data
        )

        return anno_sample

    def _write_sample(self,anno_sample, dist_path):
        assert "folder" in anno_sample
        assert "filename" in anno_sample
        assert "size" in anno_sample
        assert "width" in anno_sample["size"]
        assert "height" in anno_sample["size"]
        assert "depth" in anno_sample["size"]
        assert "object" in anno_sample

        segmented = anno_sample["segmented"] if self.include_segment and "segmented" in anno_sample else "0"
        pose = "Unspecified"

        root = Element("annotation")
        SubElement(root, 'folder').text = anno_sample["folder"]
        SubElement(root, 'filename').text = anno_sample["filename"]

        source = SubElement(root, 'source')
        SubElement(source, 'database').text = "The VOC2012 Database"
        SubElement(source, 'annotation').text = "PASCAL VOC2012"
        SubElement(source, 'image').text = "flickr"

        size = SubElement(root, 'size')
        SubElement(size, 'width').text = str(anno_sample["size"]["width"])
        SubElement(size, 'height').text = str(anno_sample["size"]["height"])
        SubElement(size, 'depth').text = str(anno_sample["size"]["depth"])

        SubElement(root, 'segmented').text = segmented

        for object in anno_sample["object"]:
            obj = SubElement(root, 'object')
            SubElement(obj, 'name').text = object["name"]
            SubElement(obj, 'pose').text = pose
            truncated = str(object["truncated"]) if "truncated" in object else "0"
            difficult = str(object["difficult"]) if "difficult" in object else "0"
            if difficult == "1" and not self.use_difficult:
                continue
            SubElement(obj, 'truncated').text = truncated
            SubElement(obj, 'difficult').text = difficult
            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(object["bndbox"]["xmin"])
            SubElement(bndbox, 'ymin').text = str(object["bndbox"]["ymin"])
            SubElement(bndbox, 'xmax').text = str(object["bndbox"]["xmax"])
            SubElement(bndbox, 'ymax').text = str(object["bndbox"]["ymax"])

        tree = ElementTree(root)
        tree.write(dist_path, encoding='utf-8', pretty_print=True)

    def _get_mask_np(self,mask_image_path, bbox):
        """
        根据mask png分割图片返回mask数组
        :param mask_image_path: 分割的mask png图片文件路径
        :param bbox: 分割框列表
        :return:
        """
        seg_image_mask = cv2.imread(mask_image_path, 0)
        mask_list = [np.expand_dims(mask_util.getsegmask(seg_image_mask, [box[0], box[1], box[2], box[3]]), 0) for
                     box in list(bbox)]
        return np.vstack(mask_list)

    def vis(self,id=None,show=False,save_path=None):
        samples = []
        if id:
            sample = self.get_sample(id)
            samples.append(sample)
        else:
            samples = self.samples

        image_vis = []
        for sample in samples:
            mask_np = None

            if sample.fields.segmented == "1":
                mask_image_path = self._seg_object_imgpath % sample.name
                mask_np = self._get_mask_np(mask_image_path, sample.bbox_list.tolist())
                mask_np[mask_np != 0] = 1

            save_path = (save_path if id else os.path.join(os.path.dirname(save_path),sample.fields.filename)) if save_path else None

            image_drawed = imshow_bboxes(sample.image,sample.bbox_list,self.categories,masks=mask_np,is_show=show,save_path=save_path)
            image_vis.append(image_drawed)

        return image_vis[0] if id else image_vis

