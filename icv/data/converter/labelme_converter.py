# -*- coding: UTF-8 -*-
from ..labelme import LabelMe
from ..coco import Coco
from ..voc import Voc
from ..core.sample import Sample
from icv.utils import reset_dir, make_empty_coco_anno, mkfile
import json
import os


class LabelMeConverter(object):
    def __init__(self, labelme):
        assert isinstance(labelme, LabelMe)
        self.labelme = labelme

    def to_coco(self, coco_root, split=None,reset=False):
        split = split if split is not None else self.labelme.split
        if reset:
            reset_dir(coco_root)
        dist_anno_path, dist_image_path = Coco.reset_dir(coco_root, split)
        dist_anno_file = os.path.join(dist_anno_path, "%s.json" % split)

        anno_data = make_empty_coco_anno()
        json.dump(anno_data, open(dist_anno_file, "w+"))

        coco = Coco(
            dist_image_path,
            dist_anno_file,
            keep_no_anno_image=self.labelme.keep_no_anno_image,
            one_index=self.labelme.one_index,
        )
        coco.set_categories(self.labelme.categories)

        image_id = 1

        for sample in self.labelme.get_samples():
            assert isinstance(sample, Sample)
            coco.sample_db[image_id] = sample
            coco.ids.append(image_id)
            image_id += 1

        coco.save(coco_root, reset_dir=True, split=split)
        coco.init()
        return coco

    def to_voc(self, voc_root, split=None,reset=False):
        split = split if split is not None else self.labelme.split
        if reset:
            reset_dir(voc_root)
        anno_path, image_path, imgset_path, imgset_seg_path, \
        seg_class_image_path, seg_object_image_path = Voc.reset_dir(voc_root)

        setfile = os.path.join(imgset_seg_path, "%s.txt" % split) if self.labelme.is_seg_mode else os.path.join(
            imgset_path, "%s.txt" % split)
        mkfile(setfile)

        voc = Voc(
            voc_root,
            split,
            keep_no_anno_image=self.labelme.keep_no_anno_image,
            mode="detect" if not self.labelme.is_seg_mode else "segment",
            categories=self.labelme.categories,
            one_index=self.labelme.one_index,
        )

        for sample in self.labelme.get_samples():
            assert isinstance(sample, Sample)
            voc.sample_db[sample.name] = sample
            voc.ids.append(sample.name)

        voc.save(voc_root, reset_dir=True, split=split)
        voc.init()
        return voc
