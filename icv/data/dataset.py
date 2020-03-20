# -*- coding: UTF-8 -*-
from ..vis.color import VIS_COLOR
from ..utils import is_seq, is_dir, mkdir, random_sample
from ..image.transforms import imcrop, imresize
from .classify import Classify
from .core import BBox, Image
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from terminaltables import AsciiTable
import shutil
import copy
import os
from time import time
from tqdm import tqdm


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
                    c: random_sample(VIS_COLOR)
                    for c in self.categories if c not in self.color_map
                })
            elif is_seq(color_list_or_map) and len(color_list_or_map) > 0:
                self.color_map = {self.categories[i]: color_list_or_map[i % len(color_list_or_map)] for i in
                                  range(len(self.categories))}
            return
        self.color_map = {}
        _colors = VIS_COLOR
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

    @property
    def is_seg_mode(self):
        if self.length == 0:
            return False

        for i in range(self.length):
            for anno in self.get_sample(self.ids[i]).annos:
                if anno.seg_mode_polys or anno.seg_mode_mask:
                    return True

        return False

    def parse_categories(self):
        categories = []
        for anno_sample in self.samples:
            label_list = [anno.label for anno in anno_sample.annos if anno.label is not None]
            if label_list:
                categories.extend(label_list)
        categories = list(set(categories))
        categories.sort()
        self.set_categories(categories)
        return categories

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
        for id in tqdm(ids):
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
            ids = random_sample(self.ids, count)
        else:
            ids = self.ids[:count]

        dataset = copy.deepcopy(self)
        dataset.ids = ids
        dataset.categories = dataset.get_categories()

        if output_dir is not None:
            dataset.save(output_dir)

        return dataset

    def copy(self):
        dataset = copy.deepcopy(self)
        return dataset

    def remove(self, ids):
        if not is_seq(ids):
            ids = [ids]

        self.ids = [_ for _ in self.ids if _ != ids]
        if hasattr(self, "sample_db"):
            for id in ids:
                del self.sample_db[id]

        return self

    def keep(self, ids):
        if not is_seq(ids):
            ids = [ids]

        self.ids = ids
        if hasattr(self, "sample_db"):
            self.sample_db = {_: self.sample_db[_] for _ in self.sample_db if _ in self.ids}

        return self

    def random(self, ratio=0, count=0):
        assert ratio > 0 or count > 0
        if ratio > 0:
            self.ids = random_sample(self.ids, ratio)
        elif count > 0:
            self.ids = random_sample(self.ids, count)

        if hasattr(self, "sample_db"):
            self.sample_db = {_: self.sample_db[_] for _ in self.sample_db if _ in self.ids}

        return self

    def divide(self, output_dir, splits=None, ratios=None):
        assert is_seq(splits) and len(splits) > 0
        assert is_seq(ratios) and len(ratios) == len(splits)
        ids = self.ids[::]
        for ix, split in enumerate(splits):
            reset_dir = True if ix == 0 else False
            _ds = self.copy()
            _ds.keep(random_sample(ids, round(ratios[ix]*len(self.ids))))
            ids = [id for id in ids if id not in _ds.ids]
            if len(ids) <= 0:
                ids = random_sample(self.ids,1)
            _ds.save(output_dir, reset_dir=reset_dir, split=split)

        return self

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
    def statistic(self, print_log=True, is_plot_show=False, plot_save_path=None):
        RATIO_TOP_K = 15
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
            "small_bbox_num": 0,
            "middle_bbox_num": 0,
            "large_bbox_num": 0,
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

        # get bbox ratio distribution
        for sample in samples:
            s = sample.statistics()
            for label in s["cats"]:
                sta["detail"][label]["num"] += s["cats"][label]

            for label in s["ratios"]:
                for r in s["ratios"][label]:
                    ratio = str(r)
                    if ratio not in sta["overview"]["ratio_distribution"]:
                        sta["overview"]["ratio_distribution"][ratio] = 0
                    sta["overview"]["ratio_distribution"][ratio] += s["ratios"][label][r]

                    if ratio not in sta["detail"][label]["ratio_distribution"]:
                        sta["detail"][label]["ratio_distribution"][ratio] = 0
                    sta["detail"][label]["ratio_distribution"][ratio] += s["ratios"][label][r]

        # truncate for top k ratio num
        sta["overview"]["ratio_distribution"] = dict(
            OrderedDict(sorted(sta["overview"]["ratio_distribution"].items(), key=lambda t: -t[1])[:RATIO_TOP_K]))

        ratio_kvs_totals = [(r, c) for r, c in sta["detail"][label]["ratio_distribution"].items() for label in
                            sta["detail"]]
        ratio_kvs = []
        forget_rks = []
        for (r, c) in ratio_kvs_totals:
            if r not in forget_rks:
                ratio_kvs.append((r, c))
            if len(ratio_kvs) >= RATIO_TOP_K:
                break
            forget_rks.append(r)

        ratio_keys = list(zip(*ratio_kvs))[0]
        for label in sta["detail"]:
            sta["detail"][label]["ratio_distribution"] = {k: sta["detail"][label]["ratio_distribution"][k] for k in
                                                          ratio_keys if k in sta["detail"][label]["ratio_distribution"]}

        '''
        ratio_kvs = []
        for i, label in enumerate(sta["detail"]):
            tmp_rd = dict(
                OrderedDict(
                    sorted(sta["detail"][label]["ratio_distribution"].items(), key=lambda t: -t[1])[:RATIO_TOP_K * 10]))

            tmp_rd_k = list(tmp_rd.keys())
            tmp_rd_v = list(tmp_rd.values())
            if i == 0:
                ratio_kvs = list(zip(tmp_rd_k, tmp_rd_v))
            else:
                ratio_kvs = [x for x in ratio_kvs if x[0] in tmp_rd_k]

        ratio_kvs = sorted(ratio_kvs, key=lambda t: -t[1])[:RATIO_TOP_K]
        ratio_keys = list(zip(*ratio_kvs))[0]

        for label in sta["detail"]:
            sta["detail"][label]["ratio_distribution"] = {k: sta["detail"][label]["ratio_distribution"][k] for k in
                                                          ratio_keys}
        '''

        # get target(bbox/seg) size distribution
        bbox_size_dict = {}
        for sample in samples:
            for anno in sample.annos:
                if anno.label not in bbox_size_dict:
                    bbox_size_dict[anno.label] = []

                if str(round(anno.bbox.ratio, 1)) in ratio_keys:
                    bbox_size_dict[anno.label].append((anno.bbox.width, anno.bbox.height))

                if anno.bbox.is_small:
                    sta["overview"]["small_bbox_num"] += 1
                elif anno.bbox.is_middle:
                    sta["overview"]["middle_bbox_num"] += 1
                elif anno.bbox.is_large:
                    sta["overview"]["large_bbox_num"] += 1

                if self.is_seg_mode:
                    if "small_segarea_num" not in sta["overview"]:
                        sta["overview"]["small_segarea_num"] = 0
                    if "middle_segarea_num" not in sta["overview"]:
                        sta["overview"]["middle_segarea_num"] = 0
                    if "large_segarea_num" not in sta["overview"]:
                        sta["overview"]["large_segarea_num"] = 0

                    if (anno.seg_mode_mask and anno.mask.is_small) or (anno.seg_mode_polys and anno.polys.is_small):
                        sta["overview"]["small_segarea_num"] += 1
                    if (anno.seg_mode_mask and anno.mask.is_middle) or (anno.seg_mode_polys and anno.polys.is_middle):
                        sta["overview"]["middle_segarea_num"] += 1
                    if (anno.seg_mode_mask and anno.mask.is_large) or (anno.seg_mode_polys and anno.polys.is_large):
                        sta["overview"]["large_segarea_num"] += 1

        if print_log or is_plot_show or plot_save_path is not None:
            # for overview
            overview_header = ["image_num", "class_num", "bbox_num", "same_image_size",
                               "min_image_width", "max_image_width", "min_image_height", "max_image_height"]
            overview_header_cn = ["图片总数", "类别总数", "目标框总数", "图片是否同尺寸",
                                  "最小图片宽度", "最大图片宽度", "最小图片高度", "最大图片高度"]
            overview_table_data = [
                sta["overview"]["image_num"], sta["overview"]["class_num"], sta["overview"]["bbox_num"],
                sta["overview"]["same_size"], sta["overview"]["min_width"], sta["overview"]["max_width"],
                sta["overview"]["min_height"], sta["overview"]["max_height"],
            ]
            overview_table = AsciiTable([overview_header, overview_header_cn, overview_table_data])
            overview_table.inner_footing_row_border = True

            # for target statistic
            target_statistic_header = ["bbox_num", "small_bbox_num", "middle_bbox_num", "large_bbox_num"]
            target_statistic_header_cn = ["目标框总数", "小目标框总数", "中目标框总数", "大目标框总数"]
            target_statistic_data = [
                sta["overview"]["bbox_num"],
                sta["overview"]["small_bbox_num"], sta["overview"]["middle_bbox_num"],
                sta["overview"]["large_bbox_num"],
            ]

            if self.is_seg_mode:
                target_statistic_header.extend(["small_segarea_num", "middle_segarea_num", "large_segarea_num"])
                target_statistic_header_cn.extend(["小目标分割区域总数", "中目标分割区域总数", "大目标分割区域总数"])
                target_statistic_data.extend([
                    sta["overview"]["small_segarea_num"], sta["overview"]["middle_segarea_num"],
                    sta["overview"]["large_segarea_num"],
                ])
            target_statistic_table = AsciiTable(
                [target_statistic_header, target_statistic_header_cn, target_statistic_data])
            target_statistic_table.inner_footing_row_border = True

            # for target rartio
            ratios = sorted(list(sta["overview"]["ratio_distribution"].keys()))

            ratio_distri_header = ["bbox ratio (h/w)"] + ratios
            ratio_distri_data = [sta["overview"]["ratio_distribution"][r] for r in ratios]
            ratio_distri_table_data = [
                ratio_distri_header,
                ["-"] + ratio_distri_data
            ]

            ratio_distri_table = AsciiTable(ratio_distri_table_data)
            ratio_distri_table.inner_footing_row_border = True

            # for categories statistic
            cats_detail_header = ["bbox ratio (h/w) \ class"] + self.categories
            cats_detail_table_data = [cats_detail_header]
            cats_num_data = [sta["detail"][cat]["num"] for cat in self.categories]
            cats_detail_table_data.append(["total num"] + cats_num_data)
            cat_ratios = sorted(list(sta["detail"][self.categories[0]]["ratio_distribution"].keys()))
            for r in cat_ratios:
                cats_detail_table_data.append(
                    [r] + [sta["detail"][cat]["ratio_distribution"].get(r, 0) for cat in self.categories])

            cats_detail_table = AsciiTable(cats_detail_table_data)
            cats_detail_table.inner_footing_row_border = False

            print("overview[概览]:")
            print(overview_table.table)

            print("\nclass detail[类别数量分布]:")
            print(cats_detail_table.table)

            print("\ntarget statistic[目标框/分割区域统计]:")
            print(target_statistic_table.table)

            print("\nbbox ratio (h/w) distribution[目标框比例(高/宽)分布]:")
            print(ratio_distri_table.table)

            if is_plot_show or plot_save_path is not None:
                import matplotlib.pyplot as plt
                fig = plt.figure(num=5, figsize=(15, 8), dpi=100)

                ax1 = fig.add_subplot(3, 2, 1)
                overview_header_post = overview_header[:3] + overview_header[4:]
                overview_table_data_post = overview_table_data[:3] + overview_table_data[4:]
                b1 = ax1.barh(list(range(len(overview_header_post))), overview_table_data_post, color='#6699CC',
                              tick_label=overview_header_post)
                for i, rect in enumerate(b1):
                    w = rect.get_width()
                    ax1.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(overview_table_data_post[i]),
                             ha='left',
                             va='center')

                ax1.set_title("overview")

                ax2 = fig.add_subplot(3, 2, 2)
                b2 = ax2.bar(list(range(len(self.categories))), cats_num_data, color="red", tick_label=self.categories)
                for xi, rect in enumerate(b2):
                    ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), "%d" % int(cats_num_data[xi]),
                             ha="center", va="bottom")

                ax2.set_title("class detail")

                ax3 = fig.add_subplot(3, 2, 3)
                b3 = ax3.barh(list(range(len(target_statistic_header))), target_statistic_data, color="green",
                              tick_label=target_statistic_header)
                for i, rect in enumerate(b3):
                    w = rect.get_width()
                    ax3.text(w, rect.get_y() + rect.get_height() / 2, '%d' % int(target_statistic_data[i]),
                             ha='left',
                             va='center')

                ax3.set_title("target statistic")

                ax4 = fig.add_subplot(3, 2, 4)
                b4 = ax4.bar(list(range(len(ratios))), ratio_distri_data, color="blue", tick_label=ratios)
                for xi, rect in enumerate(b4):
                    ax4.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), "%d" % int(ratio_distri_data[xi]),
                             ha="center", va="bottom")

                ax4.set_title("bbox ratio (h/w) distribution")

                ax5 = fig.add_subplot(3, 2, 5)
                for cat in bbox_size_dict:
                    w, h = zip(*bbox_size_dict[cat])
                    ax5.scatter(w, h, label=cat)

                ax5.set_xlabel("bbox width")
                ax5.set_ylabel("bbox height")
                ax5.set_title("bbox size distribution")
                ax5.legend()

                if is_plot_show:
                    fig.show()

                if plot_save_path is not None:
                    fig.savefig(plot_save_path)

        return sta
