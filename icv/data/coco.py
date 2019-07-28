# -*- coding: UTF-8 -*-
import os
import shutil
from .dataset import IcvDataSet
from .core.meta import SampleMeta, AnnoMeta
from .core.sample import Sample, Anno
from .core.bbox import BBox
from .core.polys import Polygon
from pycocotools.coco import COCO
from ..utils import fcopy, is_file, is_empty, is_dir, make_empty_coco_anno, json_encode, encode_to_file, \
    decode_from_file
import numpy as np
from tqdm import tqdm


class Coco(IcvDataSet):
    def __init__(self, image_dir, anno_file, split=None, keep_no_anno_image=True, one_index=True, transform=None,
                 target_transform=None):
        assert os.path.isdir(image_dir), "param image_dir is not a dir!"
        assert os.path.exists(anno_file), "param anno_file is not exist!"

        self.image_dir = image_dir
        self.anno_file = anno_file
        self.split = split if split is not None else os.path.basename(anno_file).rsplit(".", 1)[0]

        self.keep_no_anno_image = keep_no_anno_image
        self.one_index = one_index
        self.transform = transform
        self.target_transform = target_transform

        self.init()
        super(Coco, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index, False)

    def init(self):
        self.coco = COCO(self.anno_file)
        self.categories = self.get_categories()
        self.ids = self.coco.getImgIds()

        self.sample_db = {}
        self.set_colormap()

        print("there have %d samples in COCO dataset" % len(self.ids))
        print("there have %d categories in COCO dataset" % len(self.categories))

    @property
    def is_seg_mode(self):
        if self.length == 0:
            return False

        for i in range(self.length):
            for anno in self.get_sample(self.ids[i]).annos:
                if anno.seg_mode_polys or anno.seg_mode_mask:
                    return True

        return False

    def concat(self, coco, out_dir=None, reset=False, new_split=None):
        assert isinstance(coco, Coco)
        split = self.split if new_split is None else new_split
        dist_anno_path, dist_image_path = Coco.reset_dir(out_dir, split, reset=reset)
        dist_anno_file = os.path.join(dist_anno_path, "%s.json" % split)

        # TODO: concat logic
        anno_dict = decode_from_file(self.anno_file)
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

        encode_to_file(dist_anno_file, anno_dict)

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

        if reset and os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)

        dist_anno_path = os.path.join(dist_dir, "annotations")
        dist_image_path = os.path.join(dist_dir, "images", split)

        for _path in [dist_anno_path, dist_image_path]:
            if reset or not is_dir(_path):
                os.makedirs(_path)

        return dist_anno_path, dist_image_path

    def get_categories(self):
        categories = []
        catIds = sorted(self.coco.getCatIds())
        for catid in catIds:
            categories.append(self.coco.cats[catid]["name"])
        self.set_categories(categories)
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
                meta=AnnoMeta({k: ann[k] for k in ann if k not in ["bbox", "category_id", "segmentation"]})
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

    def save(self, output_dir, reset_dir=False, split=None):
        dist_anno_path, dist_image_path = Coco.reset_dir(output_dir, split, reset=reset_dir)
        dist_anno_path = os.path.join(dist_anno_path, "%s.json" % split) if is_dir(dist_anno_path) else dist_anno_path
        if is_file(self.anno_file) and not is_empty(self.image_dir):
            fcopy(self.anno_file, dist_anno_path)
            fcopy(self.image_dir, dist_image_path)
        else:
            self._write_sample(output_dir, split=split)

    def _write_sample(self, dist_dir, split=None):
        split = self.split if split is None else split
        dist_anno_path, dist_image_path = Coco.reset_dir(dist_dir, split)
        annotation = make_empty_coco_anno()

        anno_id = 1
        cats = {}
        if os.path.isdir(dist_dir):
            for id in self.ids:
                sample = self.get_sample(id)
                fcopy(sample.path, dist_image_path)
                img_height, img_width = sample.image.shape[:2]
                annotation["images"].append(
                    {
                        "id": id,
                        "file_name": os.path.basename(sample.path),
                        "width": img_width,
                        "height": img_height
                    }
                )

                for anno in sample.annos:
                    anno_dict = {}
                    anno_dict["bbox"] = [anno.bbox.xmin, anno.bbox.ymin, anno.bbox.width, anno.bbox.height]
                    if anno.seg_mode_polys:
                        anno_dict["segmentation"] = [anno.polys.exterior.flatten().tolist()]
                        anno_dict["area"] = anno.polys.area
                        anno_dict["iscrowd"] = 0
                    elif anno.seg_mode_mask:
                        anno_dict["segmentation"] = [anno.mask.to_ploygons().exterior.flatten().tolist()]
                        anno_dict["area"] = anno.mask.to_ploygons().area
                        anno_dict["iscrowd"] = 0

                    anno_dict["category_id"] = self.cat2id[anno.label]
                    anno_dict["image_id"] = id
                    anno_dict["id"] = anno_id
                    anno_id += 1

                    annotation["annotations"].append(anno_dict)

                    if anno.label not in cats:
                        cats[anno.label] = {"supercategory": "", "id": anno_dict["category_id"], "name": anno.label}

        annotation["categories"] = cats.values()
        encode_to_file(annotation, os.path.join(dist_anno_path, "%s.json" % split))

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
        for id in tqdm(self.ids):
            sample = self.get_sample(id)
            save_path = None if save_dir is None else os.path.join(save_dir, "%s.jpg" % sample.name)
            image = self.showAnns(id, with_bbox=with_bbox, with_seg=with_seg, is_show=False, save_path=save_path)
            image_vis.append(image)
        return image_vis
