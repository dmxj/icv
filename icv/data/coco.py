# -*- coding: UTF-8 -*-
import os
import json
import shutil
from .dataset import IcvDataSet
from .core.meta import SampleMeta, AnnoMeta
from .core.sample import Sample, Anno, AnnoShape
from .core.bbox import BBox
from .core.bbox_list import BBoxList
from .core.polys import Polygon
from pycocotools.coco import COCO
from icv.utils import fcopy, is_dir, is_file, is_seq_equal
from icv.image import imread, imshow, imwrite
from icv.image.vis import imdraw_polygons_with_bbox
import numpy as np


class Coco(IcvDataSet):
    def __init__(self, image_dir, anno_file, keep_no_anno_image=True, one_index=True, transform=None,
                 target_transform=None):
        assert os.path.isdir(image_dir), "param image_dir is not a dir!"
        assert os.path.exists(anno_file), "param anno_file is not exist!"

        self.image_dir = image_dir
        self.anno_file = anno_file
        self.split = os.path.basename(anno_file).rsplit(".", 1)[0]

        self.keep_no_anno_image = keep_no_anno_image
        self.coco = COCO(self.anno_file)

        # self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = target_transform

        self.categories = self.get_categories()
        self.ids = self.coco.getImgIds()

        self.sample_db = {}
        self.color_map = {}
        super(Coco, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index, False)

    def concat(self, coco, out_dir=None, reset=False, new_split=None):
        assert isinstance(coco, Coco)
        split = self.split if new_split is None else new_split
        dist_anno_path, dist_image_path = Coco.reset_dir(out_dir, split, reset=reset)
        dist_anno_file = os.path.join(dist_anno_path, "%s.json" % split)

        # TODO: concat logic
        anno_dict = json.load(open(self.anno_file, "r"))
        # copy image file
        fcopy([os.path.join(self.image_dir, self.coco.loadImgs(id)[0]['file_name']) for id in self.ids],
              dist_image_path)

        max_img_id = np.max(np.uint(self.ids))
        max_anno_id = np.max(list(self.coco.anns.keys()))
        max_cat_id = np.max(list(self.cat2id.values()))

        new_anno_id = max_anno_id + 1
        new_cat_id = max_cat_id + 1

        cat2id = dict(self.cat2id)
        image_id_map = {}
        for i, id in enumerate(coco.ids):
            sample = coco.get_sample(id)
            file_name = sample.meta.file_name
            dist_image_file = os.path.join(dist_image_path, file_name)
            new_filename = fcopy(os.path.join(coco.image_dir, file_name), dist_image_file)

            image_info = sample.meta.dict()
            image_info["id"] = max_img_id + i + 1
            image_info["file_name"] = new_filename

            anno_dict["images"].append(image_info)

            image_id_map[id] = image_info["image_id"]

            for anno in sample.annos:
                anno_meta = anno.meta.dict()
                anno_meta["id"] = new_anno_id
                new_anno_id += 1
                anno_meta["image_id"] = image_info["id"]
                if anno.label in cat2id:
                    anno_meta["category_id"] = cat2id[anno.label]
                else:
                    cat2id[anno.label] = new_cat_id
                    new_cat_id += 1
                    anno_meta["category_id"] = cat2id[anno.label]

                anno_dict["annotations"].append(anno_meta)

        anno_dict["categories"] = [{"supercategory": cat, "id": cat2id[cat], "name": cat} for cat in cat2id]

        json.dump(anno_dict, open(dist_anno_file, "w+"))

        return Coco(
            image_dir=dist_image_path,
            anno_file=dist_anno_file,
            keep_no_anno_image=self.keep_no_anno_image,
            one_index=self.one_index,
            transform=self.transform,
            target_transform=self.target_transform
        )

    @staticmethod
    def reset_dir(dist_dir, split, reset=False):
        if not reset:
            assert is_dir(dist_dir)

        if os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)

        dist_anno_path = os.path.join(dist_dir, "annotations")
        dist_image_path = os.path.join(dist_dir, "images", split)

        for _path in [dist_anno_path, dist_image_path]:
            if reset or not is_dir(_path):
                os.makedirs(_path)

        return dist_anno_path, dist_image_path

    def get_categories(self):
        categories = []
        self.id2cat = {}
        for catid in self.coco.getCatIds():
            categories.append(self.coco.cats[catid]["name"])
            self.id2cat[catid] = self.coco.cats[catid]["name"]
        self.cat2id = {self.id2cat[id]: id for id in self.id2cat}
        return categories

    def get_sample(self, id):
        if id in self.sample_db:
            return self.sample_db[id]

        path = self.coco.loadImgs(id)[0]['file_name']
        image_file = os.path.join(self.image_dir, path)

        annos = []
        for ann in self.coco.imgToAnns[id]:
            xmin, ymin, width, height = ann["bbox"]
            cat = self.id2cat[ann["category_id"]]
            annos.append(Anno(
                bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmin + width, ymax=ymin + height, label=cat),
                polys=Polygon(ann["segmentation"][0], label=cat) if "segmentation" in ann else None,
                label=cat,
                color=self.color_map[cat],
                meta=AnnoMeta({k:ann[k] for k in ann if k not in ["bbox", "category_id", "segmentation"]})
            ))

        img_info = self.coco.imgs[id]
        sample = Sample(
            name=path.rsplit(".", 1)[0],
            image=image_file,
            annos=annos,
            meta=SampleMeta(img_info)
        )
        sample.id = id
        self.sample_db[id] = sample
        return sample
        #
        # anns = self.coco.imgToAnns[id]
        # bbox_list = []
        # for ann in anns:
        #     annos.append()
        #     xmin, ymin, width, height = ann["bbox"]
        #     cat = self.id2cat[ann["category_id"]]
        #
        #
        #
        #
        #     bbox_list.append(BBox(xmin, ymin, xmin + width, ymin + height, label=cat, **ann))
        #
        # img_info = self.coco.imgs[id]
        # sample = Sample.init(path.rsplit(".", 1)[0], BBoxList(bbox_list=bbox_list), image_file, **img_info)
        # sample.id = id
        # sample.add_field("anns", anns)
        #
        # self.sample_db[id] = sample
        # return sample

    def write_sample(self, dist_dir):
        dist_anno_path, dist_image_path = Coco.reset_dir(dist_dir, self.split)
        annotation = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": [],
        }

        anno_id = 1
        cats = {}
        if os.path.isdir(dist_dir):
            for id in self.ids:
                sample = self.get_sample(id)
                imwrite(sample.image, os.path.join(dist_image_path, sample.meta.file_name))
                img_height, img_width = sample.image.shape[:2]
                annotation["images"].append(
                    {
                        "id": id,
                        "file_name": sample.meta.file_name,
                        "width": img_width,
                        "height": img_height
                    }
                )

                for anno in sample.annos:
                    anno_dict = anno.meta.dict()
                    anno_dict["bbox"] = [anno.bbox.xmin, anno.bbox.ymin, anno.bbox.width, anno.bbox.height]
                    if anno.seg_mode_polys:
                        anno_dict["segmentation"] = [anno.polys.exterior.flatten().tolist()]
                    elif anno.seg_mode_mask:
                        anno_dict["segmentation"] = [anno.mask.to_ploygons().exterior.flatten().tolist()]

                    anno_dict["category_id"] = self.cat2id[anno.label]
                    anno_dict["image_id"] = id
                    anno_dict["id"] = anno_id
                    anno_id += 1

                    annotation["annotations"].append(anno_dict)

                    if anno.label not in cats:
                        cats[anno.label] = {"supercategory": "", "id": anno["category_id"], "name": anno.label}

        annotation["categories"] = cats.values()
        json.dump(annotation, open(os.path.join(dist_anno_path, "%s.json" % self.split), "w"))

    # def _write_sample(self, anno_samples, dist_path):
    #     annotation = {
    #         "images": [],
    #         "annotations": [],
    #         "categories": [],
    #     }
    #
    #     anno_id = 0
    #     cats = {}
    #     for id, sample in enumerate(anno_samples):
    #         if sample.has_field("file_name"):
    #             continue
    #         img_height, img_width = sample.image.shape[:2]
    #         annotation["images"].append(
    #             {
    #                 "id": id,
    #                 "file_name": sample.get_field("file_name"),
    #                 "width": img_width,
    #                 "height": img_height
    #             }
    #         )
    #
    #         for bbox in sample.bbox_list:
    #             if bbox.has_field("label"):
    #                 continue
    #             anno = {
    #                 "bbox": [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
    #             }
    #
    #             if bbox.has_field("segmentation"):
    #                 anno["segmentation"] = bbox.get_field("segmentation")
    #
    #             if bbox.has_field("area"):
    #                 anno["area"] = bbox.get_field("area")
    #
    #             if bbox.has_field("iscrowd"):
    #                 anno["iscrowd"] = bbox.get_field("iscrowd")
    #
    #             anno["category_id"] = self.cat2id[bbox.lable]
    #             anno["image_id"] = id
    #             anno["id"] = anno_id
    #             anno_id += 1
    #
    #             if bbox.lable not in cats:
    #                 cats[bbox.lable] = {"supercategory": "", "id": anno["category_id"], "name": bbox.lable}
    #
    #     annotation["categories"] = cats.values()
    #     json.dump(annotation, open(dist_path, "w"))

    def showAnns(self, id, with_bbox=True, with_seg=True, is_show=True, save_path=None):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        sample = self.get_sample(id)
        return sample.vis(with_bbox=with_bbox, with_seg=with_seg, is_show=is_show, save_path=save_path)
        # polygons = []
        # name_list = []
        # for ann in sample.fields.anns:
        #     if "segmentation" in ann:
        #         segmentation = ann["segmentation"]
        #         polygons.append(list(zip(segmentation[0][0::2], segmentation[0][1::2])))
        #     elif "bbox" in ann and len(ann["bbox"]) == 4:
        #         bbox = ann["bbox"]
        #         xmin, ymin, width, height = bbox
        #         polygons.append((xmin, ymin))
        #         polygons.append((xmin + width, ymin))
        #         polygons.append((xmin + width, ymin + height))
        #         polygons.append((xmin, ymin + height))
        #     name_list.append(self.id2cat[ann["category_id"]])
        #
        # image = imdraw_polygons_with_bbox(sample.image, polygons, name_list, with_bbox=with_bbox,
        #                                   color_map=self.color_map, save_path=save_path)
        # if is_show:
        #     imshow(image)
        #
        # return image

    def vis(self, id=None, with_bbox=True, with_seg=True, is_show=False, save_dir=None, reset_dir=False):
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            elif reset_dir:
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)

        if id is not None:
            sample = self.get_sample(id)
            save_path = None if save_dir is None else os.path.join(save_dir, "%s.jpg" % sample.name)
            return self.showAnns(id, with_bbox=with_bbox, with_seg=with_seg, is_show=is_show, save_path=save_path)

        image_vis = []
        for id in self.ids:
            sample = self.get_sample(id)
            save_path = None if save_dir is None else os.path.join(save_dir, "%s.jpg" % sample.name)
            image = self.showAnns(id, with_bbox=with_bbox, with_seg=with_seg, is_show=False, save_path=save_path)
            image_vis.append(image)
        return image_vis
