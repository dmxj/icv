# -*- coding: UTF-8 -*-
from ..vis.color import STANDARD_COLORS
from ..utils import is_seq, is_dir, mkdir
from ..image.transforms import imcrop, imresize
from .classify import Classify
from .core import BBox, Image
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from terminaltables import AsciiTable
import shutil
import random
import copy
import os
from time import time


class IcvDataSet(object):
    __metaclass__ = ABCMeta

    def __init__(self, ids=None, categories=None, keep_no_anno_image=True, one_index=False, build_cat=True):
        self.ids = ids if ids else []
        self.categories = categories if categories else []
        self.keep_no_anno_image = keep_no_anno_image
        self.one_index = one_index
        self.num_classes = len(self.categories)

        if build_cat:
            self._build()

        if not hasattr(self, "color_map") or self.color_map is None or len(self.color_map) == 0:
            self.set_colormap()

    def _build(self):
        if self.one_index:
            self.id2cat = {i + 1: cat for i, cat in enumerate(self.categories)}
            self.cat2id = {cat: i + 1 for i, cat in enumerate(self.categories)}
        else:
            self.id2cat = {i: cat for i, cat in enumerate(self.categories)}
            self.cat2id = {cat: i for i, cat in enumerate(self.categories)}

        self.id2index = {id: i for i, id in enumerate(self.ids)}
        self.index2id = {i: id for i, id in enumerate(self.ids)}

    def get_img_info(self, index):
        img_id = self.ids[index]
        return img_id

    def set_colormap(self, color_list_or_map=None):
        assert color_list_or_map is None or isinstance(color_list_or_map, dict) or is_seq(color_list_or_map)
        if color_list_or_map is not None and len(self.categories) > 0:
            if isinstance(color_list_or_map, dict):
                self.color_map = {c: color_list_or_map[c] for c in color_list_or_map if c in self.categories}
                self.color_map.update({
                    c: random.choice(STANDARD_COLORS)
                    for c in self.categories if c not in self.color_map
                })
            elif is_seq(color_list_or_map) and len(color_list_or_map) > 0:
                self.color_map = {self.categories[i]: color_list_or_map[i % len(color_list_or_map)] for i in
                                  range(len(self.categories))}
            return
        self.color_map = {}
        _colors = STANDARD_COLORS
        # random.shuffle(_colors)
        for i, cat in enumerate(self.categories):
            self.color_map[cat] = _colors[i % len(_colors)]

    @property
    def length(self):
        return len(self)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.get_sample(self.index2id[item])

    def get_categories(self):
        return self.categories

    def set_categories(self, categories):
        assert is_seq(categories)
        self.categories = list(categories)

        self.id2cat = {}
        self.cat2id = {}
        for i, cate in enumerate(self.categories):
            if self.one_index:
                self.id2cat[i + 1] = cate
                self.cat2id[cate] = i + 1
            else:
                self.id2cat[i] = cate
                self.cat2id[cate] = i

    def get_samples(self, ids=None):
        self.samples = []
        ids = ids if ids is not None else self.ids
        for id in ids:
            sample = self.get_sample(id)
            if sample.count == 0 and not self.keep_no_anno_image:
                continue
            self.samples.append(sample)

        return self.samples

    def get_sample(self, id):
        pass

    def get_groundtruth(self, id):
        return self.get_sample(id)

    @abstractmethod
    def vis(self, save_dir=None):
        pass

    @abstractmethod
    def sub(self, count=0, ratio=0, shuffle=True, output_dir=None):
        if count <= 0:
            count = round(self.length * ratio)

        count = min(self.length, count)
        if shuffle:
            ids = random.sample(self.ids, count)
        else:
            ids = self.ids[:count]

        dataset = copy.deepcopy(self)
        dataset.ids = ids
        dataset.categories = dataset.get_categories()

        if output_dir is not None:
            dataset.save(output_dir)

        return dataset

    @abstractmethod
    def save(self, output_dir, reset_dir=False, split=None):
        pass

    @abstractmethod
    def concat(self, dataset, out_dir, reset=False, new_split=None):
        pass

    @abstractmethod
    def crop_bbox_for_classify(self, output_dir, reset=False, split="train", pad=0, resize=None):
        assert resize is None or (is_seq(resize) and len(resize) == 2)
        if reset and is_dir(output_dir):
            shutil.rmtree(output_dir)
        if not is_dir(output_dir):
            mkdir(output_dir)
        samples = self.get_samples()

        tmp_dir = os.path.join(output_dir, str(time()))
        mkdir(tmp_dir)

        classify_dataset = Classify(root_dir=output_dir, categories=self.categories)
        for sample in samples:
            _bboxes = []
            _cats = []
            for anno in sample.annos:
                if isinstance(anno.bbox, BBox):
                    _bbox = anno.bbox.copy()
                    _bbox.extend(pad, sample.shape)
                    _bboxes.append(_bbox.bbox)
                    _cats.append(anno.label)
            patches = imcrop(sample.image, _bboxes)
            assert len(patches) == len(_cats)

            for i, p in enumerate(patches):
                if resize is not None:
                    p = imresize(p, resize)
                imgobj = Image(img=p, ext=sample.ext, name="")
                classify_dataset.img2cat[imgobj] = _cats[i]
                classify_dataset.img2set[imgobj] = split

        classify_dataset.save(output_dir, rename=True)
        shutil.rmtree(tmp_dir)

    @abstractmethod
    def statistic(self, print_log=False):
        samples = self.get_samples()

        sta = dict(overview=dict(), detail=dict())
        sta["overview"] = {
            "image_num": self.length,
            "class_num": len(self.categories),
            "bbox_num": sum([sample.count for sample in samples]),
            "min_width": min([sample.width for sample in samples]),
            "max_width": max([sample.width for sample in samples]),
            "min_height": min([sample.height for sample in samples]),
            "max_height": max([sample.height for sample in samples]),
            "same_size": len(set([(sample.height, sample.width) for sample in samples])) == 1,
            "ratio_distribution": OrderedDict(),
        }

        sta["detail"] = {
            cat: {
                "num": 0,
                "ratio_distribution": OrderedDict(),
            }
            for cat in self.categories
        }

        for sample in samples:
            s = sample.statistics()
            for label in s["cats"]:
                sta["detail"][label]["num"] = s["cats"][label]

            for label in s["ratios"]:
                for r in s["ratios"][label]:
                    ratio = str(r)
                    if ratio not in sta["overview"]["ratio_distribution"]:
                        sta["overview"]["ratio_distribution"][ratio] = 0
                    sta["overview"]["ratio_distribution"][ratio] += s["ratios"][label][r]

                    if ratio not in sta["detail"][label]["ratio_distribution"]:
                        sta["detail"][label]["ratio_distribution"][ratio] = 0
                    sta["detail"][label]["ratio_distribution"][ratio] += s["ratios"][label][r]

        if print_log:
            overview_header = ["image_num", "class_num", "bbox_num", "same_image_size",
                               "min_image_width", "max_image_width", "min_image_height", "max_image_height"]
            overview_table_data = [
                overview_header,
                [
                    sta["overview"]["image_num"], sta["overview"]["class_num"], sta["overview"]["bbox_num"],
                    sta["overview"]["same_size"], sta["overview"]["min_width"], sta["overview"]["max_width"],
                    sta["overview"]["min_height"], sta["overview"]["max_height"],
                ]
            ]
            overview_table = AsciiTable(overview_table_data)
            overview_table.inner_footing_row_border = True

            ratios = sorted(list(sta["overview"]["ratio_distribution"].keys()))

            ratio_distri_header = ["bbox ratio (h/w)"] + ratios
            ratio_distri_table_data = [
                ratio_distri_header,
                ["-"] + [sta["overview"]["ratio_distribution"][r] for r in ratios]
            ]

            ratio_distri_table = AsciiTable(ratio_distri_table_data)
            ratio_distri_table.inner_footing_row_border = True

            cats_detail_header = ["bbox ratio (h/w) \ class"] + self.categories
            cats_detail_table_data = [cats_detail_header]
            cats_detail_table_data.append(["num"] + [sta["detail"][cat]["num"] for cat in self.categories])
            for r in ratios:
                cats_detail_table_data.append(
                    [r] + [sta["detail"][cat]["ratio_distribution"].get(r, 0) for cat in self.categories])

            cats_detail_table = AsciiTable(cats_detail_table_data)
            cats_detail_table.inner_footing_row_border = True

            print("overview:")
            print(overview_table.table)

            print("\nratio distribution:")
            print(ratio_distri_table.table)

            print("\nclass detail:")
            print(cats_detail_table.table)

        return sta
