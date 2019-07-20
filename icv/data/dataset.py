# -*- coding: UTF-8 -*-
from icv.vis.color import MASK_COLORS
from abc import ABC, ABCMeta, abstractmethod
import random
import copy


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

    def set_colormap(self):
        self.color_map = {}
        _colors = MASK_COLORS
        random.shuffle(_colors)
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
        pass

    def get_samples(self, ids=None):
        self.samples = []
        ids = ids if ids is not None else self.ids
        for id in ids:
            sample = self.get_sample(id)
            if sample.bbox_list.length == 0 and not self.keep_no_anno_image:
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
    def save(self, output_dir, split=None):
        pass

    @abstractmethod
    def concat(self, dataset, out_dir, reset=False, new_split=None):
        pass

    @abstractmethod
    def statistic(self):
        pass
