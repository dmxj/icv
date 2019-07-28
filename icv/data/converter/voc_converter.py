# -*- coding: UTF-8 -*-
from ..voc import Voc
from ..coco import Coco
from ..core.sample import Sample
from icv.utils import reset_dir,make_empty_coco_anno
import json
import os

class VocConverter(object):
    def __init__(self,voc):
        assert isinstance(voc,Voc)
        self.voc = voc

    def to_coco(self,coco_root,split=None,reset=False):
        split = split if split is not None else self.voc.split
        if reset:
            reset_dir(coco_root)
        dist_anno_path, dist_image_path = Coco.reset_dir(coco_root,split)
        dist_anno_file = os.path.join(dist_anno_path,"%s.json" % split)

        anno_data = make_empty_coco_anno()
        json.dump(anno_data,open(dist_anno_file,"w+"))

        coco = Coco(
                    dist_image_path,
                    dist_anno_file,
                    keep_no_anno_image=self.voc.keep_no_anno_image,
                    one_index=self.voc.one_index,
                )
        coco.set_categories(self.voc.categories)

        image_id = 1

        for sample in self.voc.get_samples():
            assert isinstance(sample,Sample)
            coco.sample_db[image_id] = sample
            coco.ids.append(image_id)
            image_id += 1

        coco.save(coco_root,reset_dir=True,split=split)
        coco.init()
        return coco


