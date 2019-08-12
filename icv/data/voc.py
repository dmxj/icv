# -*- coding: UTF-8 -*-
from .dataset import IcvDataSet
import os
import shutil
import numpy as np
from ..utils import load_voc_anno, save_voc_anno, make_empty_voc_anno, fcopy, list_from_file, list_to_file, is_file, \
    is_dir, mkdir
from ..image import imread
from ..data.core.meta import SampleMeta, AnnoMeta
from ..data.core.sample import Sample, Anno
from ..data.core.bbox import BBox
from ..data.core.mask import Mask
from ..data.core.polys import Polygon
from tqdm import tqdm
import random
from ..vis.color import STANDARD_COLORS


# TODO: 分割数据（图像目录、segment=1）
class Voc(IcvDataSet):
    def __init__(self, data_dir, split=None, use_difficult=True, keep_no_anno_image=True, mode="detect",
                 include_segment=True, categories=None, one_index=False, image_suffix="jpg"):
        self.root = data_dir
        self.split = split if split else "trainval"
        self.mode = mode
        self.is_seg_mode = self.mode != "detect"
        self.include_segment = include_segment
        self.image_set = "Main" if self.mode == "detect" else "Segmentation"

        self.use_difficult = use_difficult
        self.keep_no_anno_image = keep_no_anno_image
        self.image_suffix = image_suffix
        self.one_index = one_index

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s." + image_suffix)
        self._seg_class_imgpath = os.path.join(self.root, "SegmentationClass", "%s.png")
        self._seg_object_imgpath = os.path.join(self.root, "SegmentationObject", "%s.png")
        self._imgsetpath = os.path.join(self.root, "ImageSets", self.image_set, "%s.txt")

        self.categories = categories

        self.init()

        super(Voc, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index)

    def init(self):
        self.ids = list_from_file(self._imgsetpath % self.split)
        self.id2img = {k: v for k, v in enumerate(self.ids)}

        self.sample_db = {}
        self.color_map = {}
        if self.categories is None:
            print("parsing categories ...")
            self.categories = self.get_categories()

        self.set_categories(self.categories)
        self.set_colormap(self.color_map)
        print("there have %d samples in VOC dataset" % len(self.ids))
        print("there have %d categories in VOC dataset" % len(self.categories))

    def concat(self, voc, output_dir, reset=False, new_split=None):
        assert isinstance(voc, Voc)
        if new_split is None:
            if self.split == voc.split:
                new_split = self.split
            else:
                new_split = self.split + voc.split

        anno_path, image_path, imgset_path, imgset_seg_path, seg_class_image_path, seg_object_image_path = Voc.reset_dir(
            output_dir, reset=reset)

        ps0 = fcopy([self._imgpath % id for id in self.ids], image_path)
        ps1 = fcopy([self._imgpath % id for id in voc.ids], image_path)

        ids0 = [os.path.basename(_).rsplit(".", 1)[0] for _ in ps0]
        ids1 = [os.path.basename(_).rsplit(".", 1)[0] for _ in ps1]

        fcopy([self._annopath % id for id in self.ids], anno_path)
        fcopy([voc._annopath % id for id in voc.ids], anno_path)

        if self.is_seg_mode:
            fcopy([self._seg_class_imgpath % id for id in self.ids], seg_class_image_path)
            fcopy([self._seg_object_imgpath % id for id in self.ids], seg_object_image_path)

        if voc.is_seg_mode:
            fcopy([voc._seg_class_imgpath % id for id in voc.ids], seg_class_image_path)
            fcopy([voc._seg_object_imgpath % id for id in voc.ids], seg_object_image_path)

        if self.mode == voc.mode:
            ids = ids0 + ids1
            setpath = imgset_seg_path if self.is_seg_mode else imgset_path

            list_to_file(ids, os.path.join(setpath, "%s.txt" % new_split))
        else:
            if not self.is_seg_mode:
                list_to_file(ids0, os.path.join(imgset_path, "%s.txt" % new_split))
            else:
                list_to_file(ids0, os.path.join(imgset_seg_path, "%s.txt" % new_split))

            if not voc.is_seg_mode:
                list_to_file(ids1, os.path.join(imgset_path, "%s.txt" % new_split))
            else:
                list_to_file(ids1, os.path.join(imgset_seg_path, "%s.txt" % new_split))

        return Voc(
            output_dir,
            split=new_split,
            use_difficult=self.use_difficult,
            keep_no_anno_image=self.keep_no_anno_image,
            mode=self.mode,
            include_segment=self.include_segment,
            one_index=self.one_index,
            categories=self.categories + voc.categories,
            image_suffix=self.image_suffix
        )

    def sub(self, count=0, ratio=0, shuffle=True, output_dir=None):
        voc = super(Voc, self).sub(count, ratio, shuffle, output_dir)
        voc.root = output_dir
        return voc

    def save(self, output_dir, reset_dir=False, split=None):
        split = split if split is not None else self.split
        anno_path, image_path, imgset_path, imgset_seg_path, seg_class_image_path, seg_object_image_path = Voc.reset_dir(
            output_dir, reset=reset_dir)
        for id in self.ids:
            if is_file(self._annopath % id) and is_file(self._imgpath % id):
                fcopy(self._annopath % id, anno_path)
                fcopy(self._imgpath % id, image_path)
                fcopy(self._seg_class_imgpath % id, seg_class_image_path)
                fcopy(self._seg_object_imgpath % id, seg_object_image_path)
            else:
                self._write_sample(self.get_sample(id), output_dir)

        if self.is_seg_mode:
            list_to_file(self.ids, os.path.join(imgset_seg_path, "%s.txt" % split))
        else:
            list_to_file(self.ids, os.path.join(imgset_path, "%s.txt" % split))

    @staticmethod
    def reset_dir(dist_dir, reset=False):
        if not reset:
            assert is_dir(dist_dir)
        if reset and os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)

        anno_path = os.path.join(dist_dir, "Annotations")
        image_path = os.path.join(dist_dir, "JPEGImages")
        imgset_path = os.path.join(dist_dir, "ImageSets", "Main")

        imgset_seg_path = os.path.join(dist_dir, "ImageSets", "Segmentation")
        seg_class_image_path = os.path.join(dist_dir, "SegmentationClass")
        seg_object_image_path = os.path.join(dist_dir, "SegmentationObject")

        for _path in [anno_path, image_path, imgset_path, imgset_seg_path, seg_class_image_path, seg_object_image_path]:
            if reset or not is_dir(_path):
                mkdir(_path)

        return anno_path, image_path, imgset_path, imgset_seg_path, seg_class_image_path, seg_object_image_path

    def get_categories(self):
        self.get_samples()
        categories = []
        for anno_sample in self.samples:
            label_list = [l.label for l in anno_sample.annos if l.label is not None]
            if label_list:
                categories.extend(label_list)
        categories = list(set(categories))
        categories.sort()
        return categories

    def get_sample(self, id):
        if id in self.sample_db:
            return self.sample_db[id]

        anno_file = self._annopath % id
        image_file = self._imgpath % id

        if not is_file(image_file):
            raise FileNotFoundError("image file : {} not exist!".format(image_file))

        if not is_file(anno_file):
            anno_data = make_empty_voc_anno()
        else:
            anno_data = load_voc_anno(anno_file)["annotation"]

        sample_meta = SampleMeta({
            k: anno_data[k]
            for k in anno_data if k not in ["object"]
        })

        segcls_file = self._seg_class_imgpath % id
        segobj_file = self._seg_object_imgpath % id

        img_segcls = None
        if is_file(segcls_file):
            img_segcls = imread(segcls_file, 0)

        img_segobj = None
        if is_file(segobj_file):
            img_segobj = imread(segobj_file, 0)

        annos = []
        if "object" in anno_data:
            for obj in anno_data["object"]:
                if "difficult" in obj and obj["difficult"] != '0' and self.use_difficult:
                    continue
                label = obj["name"]
                if label not in self.color_map:
                    self.color_map[label] = random.choice(STANDARD_COLORS)

                xmin = int(obj["bndbox"]["xmin"])
                ymin = int(obj["bndbox"]["ymin"])
                xmax = int(obj["bndbox"]["xmax"])
                ymax = int(obj["bndbox"]["ymax"])
                anno = Anno(
                    bbox=BBox(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        label=label,
                    ),
                    label=label,
                    color=self.color_map[label],
                    meta=AnnoMeta({
                        k: obj[k]
                        for k in obj if k not in ["name", "bndbox"]
                    })
                )

                if img_segobj is not None:
                    bin_mask = np.zeros_like(img_segobj, np.uint8)
                    bin_mask[ymin:ymax, xmin:xmax] = img_segobj[ymin:ymax, xmin:xmax]
                    bin_mask[np.where(bin_mask != 0)] = 1
                    anno.mask = Mask(bin_mask)
                elif img_segcls is not None:
                    bin_mask = np.zeros_like(img_segcls, np.uint8)
                    bin_mask[ymin:ymax, xmin:xmax] = img_segcls[ymin:ymax, xmin:xmax]
                    bin_mask[np.where(bin_mask != 0)] = 1
                    anno.mask = Mask(bin_mask)

                annos.append(anno)

        sample = Sample(
            id,
            image_file,
            annos,
            sample_meta
        )

        self.sample_db[id] = sample
        return sample

    def _write_sample(self, sample, dist_dir):
        assert isinstance(sample, Sample)

        reset_dir = not os.path.exists(dist_dir)
        anno_path, image_path, imgset_path, imgset_seg_path, \
        seg_class_image_path, seg_object_image_path = Voc.reset_dir(dist_dir, reset_dir)

        anno_dist_path = os.path.join(anno_path, "%s.xml" % sample.name)
        anno_data = {
            "segmented": 1 if self.is_seg_mode else 0,
            "folder": image_path,
            "filename": os.path.basename(sample.path),
            "size": {
                "width": sample.width,
                "height": sample.height,
                "depth": sample.dim
            },
            "objects": []
        }

        for ix, anno in enumerate(sample.annos):
            bndbox = anno.meta.dict()
            if "truncated" not in bndbox and "difficult" not in bndbox:
                bndbox = {
                    "truncated": 1,
                    "difficult": 0,
                }
            bndbox["name"] = anno.label
            bndbox["bndbox"] = {
                "xmin": anno.bbox.xmin_int,
                "ymin": anno.bbox.ymin_int,
                "xmax": anno.bbox.xmax_int,
                "ymax": anno.bbox.ymax_int,
            }

            catid = self.cat2id[anno.label]
            if not self.one_index:
                catid += 1

            if anno.seg_mode_mask:
                anno.mask.save(os.path.join(seg_class_image_path, "%s.png" % sample.name), id=catid)
                # TODO: make sure object segment mask id
                anno.mask.save(os.path.join(seg_object_image_path, "%s.png" % sample.name), id=ix + 1)
            elif anno.seg_mode_polys:
                assert isinstance(anno.polys, Polygon)
                anno.polys.to_mask(sample.height, sample.width).save(
                    os.path.join(seg_class_image_path, "%s.png" % sample.name),
                    id=catid)
                # TODO: make sure object segment mask id
                anno.polys.to_mask(sample.height, sample.width).save(
                    os.path.join(seg_object_image_path, "%s.png" % sample.name), id=ix + 1)

            anno_data["objects"].append(bndbox)

        save_voc_anno(anno_data, anno_dist_path)
        fcopy(sample.path, image_path)

    def vis(self, id=None, with_bbox=True, with_seg=True, is_show=False, save_dir=None, reset_dir=False):
        if save_dir is not None:
            if not os.path.exists(save_dir):
                mkdir(save_dir)
            elif reset_dir:
                shutil.rmtree(save_dir)
                mkdir(save_dir)

        if id is not None:
            sample = self.get_sample(id)
            save_path = None if save_dir is None else os.path.join(save_dir, "%s.jpg" % sample.name)
            return sample.vis(with_bbox=with_bbox, with_seg=with_seg, is_show=is_show, save_path=save_path)

        image_vis = []
        for id in tqdm(self.ids):
            sample = self.get_sample(id)
            save_path = None if save_dir is None else os.path.join(save_dir, "%s.jpg" % sample.name)
            image = sample.vis(with_bbox=with_bbox, with_seg=with_seg, is_show=is_show, save_path=save_path)
            image_vis.append(image)
        return image_vis
