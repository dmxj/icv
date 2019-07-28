# -*- coding: UTF-8 -*-
from .dataset import IcvDataSet
from ..utils import is_seq, is_dir
from ..image import imwrite
from ..data.core.bbox import BBox
from ..data.core.polys import Polygon
from ..data.core.sample import Sample, Anno
from ..data.core.meta import AnnoMeta,SampleMeta
from ..vis.color import STANDARD_COLORS
import random
import os
import json
import shutil
from tqdm import tqdm


class LabelMe(IcvDataSet):
    def __init__(self, image_anno_path_list, split="trainval", keep_no_anno_image=True, categories=None,
                 one_index=False):
        assert is_seq(image_anno_path_list)
        image_anno_path_list = list(image_anno_path_list)
        image_path_list, anno_path_list = list(zip(*image_anno_path_list))

        self.split = split
        self.keep_no_anno_image = keep_no_anno_image
        self.one_index = one_index

        self.ids = [os.path.basename(_).rsplit(".", 1)[0] for _ in image_path_list]
        self.id2imgpath = {id: image_path_list[ix] for ix, id in enumerate(self.ids)}
        self.id2annopath = {id: anno_path_list[ix] for ix, id in enumerate(self.ids)}

        self.sample_db = {}
        self.color_map = {}
        self.get_samples()
        self.categories = categories if categories is not None else self.get_categories()

        super(LabelMe, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index)
        print("there have %d samples in LabelMe dataset" % len(self.ids))
        print("there have %d categories in LabelMe dataset" % len(self.categories))

    @property
    def is_seg_mode(self):
        if self.length == 0:
            return False

        for i in range(self.length):
            for anno in self.get_sample(self.ids[i]).annos:
                if anno.seg_mode_polys or anno.seg_mode_mask:
                    return True

        return False

    def get_categories(self):
        categories = []
        for anno_sample in self.samples:
            label_list = [anno.label for anno in anno_sample.annos]
            if label_list:
                categories.extend(label_list)
        categories = list(set(categories))
        categories.sort()
        self.set_categories(categories)
        return categories

    def save(self, output_dir, reset_dir=False, split=None):
        anno_path, image_path = LabelMe.reset_dir(output_dir, reset=reset_dir)
        for id in self.ids:
            self._write(self.get_sample(id), anno_path, image_path)

    @staticmethod
    def reset_dir(dist_dir, reset=False):
        if not reset:
            assert is_dir(dist_dir)
        if reset and os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)

        anno_path = os.path.join(dist_dir, "annotations")
        image_path = os.path.join(dist_dir, "images")

        for _path in [anno_path, image_path]:
            if reset or not is_dir(_path):
                os.makedirs(_path)

        return anno_path, image_path

    def _get_bbox_from_points(self, points):
        """
        根据polygon顶点获取bbox
        :param points:
        :return:
        """
        x_list = [p[0] for p in points]
        y_list = [p[1] for p in points]

        xmin = min(x_list)
        ymin = min(y_list)
        xmax = max(x_list)
        ymax = max(y_list)

        return xmin, ymin, xmax, ymax

    def get_sample(self, id):
        """
        get sample
        :param id: image name
        :return:
        """
        if id in self.sample_db:
            return self.sample_db[id]

        anno_file = self.id2annopath[id]
        anno_data = json.load(open(anno_file, "r"))
        img_file = self.id2imgpath[id]

        annos = []
        if "shapes" in anno_data:
            shapes = anno_data["shapes"]
            for shape in shapes:
                if "shape_type" not in shape or "points" not in shape or "label" not in shape:
                    continue

                points = shape["points"]
                xmin, ymin, xmax, ymax = self._get_bbox_from_points(points)

                label = shape["label"]
                if label not in self.color_map:
                    self.color_map[label] = random.choice(STANDARD_COLORS)

                anno = Anno(
                    bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label),
                    label=label,
                    color=self.color_map[label],
                    polys=Polygon.init_from(points, label=label) if shape["shape_type"] == "polygon" else None,
                    meta=AnnoMeta()

                )
                annos.append(anno)

        sample = Sample(
            name=id,
            image=img_file,
            annos=annos,
            meta=SampleMeta()
        )
        return sample

    def _write(self, anno_sample, anno_path, img_path):
        assert isinstance(anno_sample, Sample)

        if is_dir(anno_path):
            anno_path = os.path.join(anno_path, "%s.json" % anno_sample.name)

        if is_dir(img_path):
            img_path = os.path.join(img_path, "%s.jpg" % anno_sample.name)

        imwrite(anno_sample.image, img_path)
        anno_json = {
            "shapes": [],
            "imagePath": img_path,
            "imageHeight": anno_sample.height,
            "imageWidth": anno_sample.width
        }

        for anno in anno_sample.annos:
            shape = {
                "label": anno.label,
                "shape_type": "polygon" if anno.seg_mode else "rectangle"
            }

            if anno.seg_mode_polys:
                shape["points"] = anno.polys.exterior.tolist()
            elif anno.seg_mode_mask:
                shape["points"] = anno.mask.to_ploygons().exterior.tolist()
            else:
                shape["points"] = [[anno.bbox.xmin, anno.bbox.ymin], [anno.bbox.xmax, anno.bbox.ymax]]

            anno_json["shapes"].append(shape)

        json.dump(anno_json, open(anno_path, "w"))

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
            return sample.vis(with_bbox=with_bbox, with_seg=with_seg, is_show=is_show, save_path=save_path)

        image_vis = []
        for id in tqdm(self.ids):
            sample = self.get_sample(id)
            save_path = None if save_dir is None else os.path.join(save_dir, "%s.jpg" % sample.name)
            image = sample.vis(with_bbox=with_bbox, with_seg=with_seg, is_show=False, save_path=save_path)
            image_vis.append(image)
        return image_vis
