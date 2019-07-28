# -*- coding: UTF-8 -*-
from ..voc import Voc
from ..coco import Coco
from ..core.sample import Sample
from icv.utils import reset_dir, mkfile
import os


class CocoConverter(object):
    def __init__(self, coco):
        assert isinstance(coco, Coco)
        self.coco = coco

    def to_voc(self, voc_root, split=None,reset=False):
        split = split if split is not None else self.coco.split
        if reset:
            reset_dir(voc_root)
        anno_path, image_path, imgset_path, imgset_seg_path, \
        seg_class_image_path, seg_object_image_path = Voc.reset_dir(voc_root)

        setfile = os.path.join(imgset_seg_path, "%s.txt" % split) if self.coco.is_seg_mode else os.path.join(
            imgset_path, "%s.txt" % split)
        mkfile(setfile)

        voc = Voc(
            voc_root,
            split,
            keep_no_anno_image=self.coco.keep_no_anno_image,
            mode="detect" if not self.coco.is_seg_mode else "segment",
            categories=self.coco.categories,
            one_index=self.coco.one_index,
        )

        for sample in self.coco.get_samples():
            assert isinstance(sample, Sample)
            voc.sample_db[sample.name] = sample
            voc.ids.append(sample.name)

        voc.save(voc_root, reset_dir=True, split=split)
        voc.init()
        return voc
