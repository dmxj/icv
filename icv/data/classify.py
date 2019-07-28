# -*- coding: UTF-8 -*-
from ..utils import is_dir, list_from_file
from ..image.io import imwrite
from .core import Image
import os
import shutil

class Classify(object):
    def __init__(self, root_dir=None, sample_file_list=None, sample_split="\t", categories=None, one_index=False):
        assert is_dir(root_dir) or sample_file_list is not None

        if is_dir(root_dir):
            self.root_dir = root_dir
            self.img2cat, self.img2set = Classify.extract_by_dir(self.root_dir)
        else:
            self.img2cat = {}
            self.img2set = {}
            self.categories = []

            for sample_file in sample_file_list:
                img2cat, img2set = Classify.extract_by_samplefile(sample_file, sample_split)

                self.img2cat.update(img2cat)
                self.img2set.update(img2set)

        if categories is not None:
            self.categories = categories
        else:
            self.categories = list(set(self.img2cat.values()))

    def save(self, output, rename=False, suffix=None):
        catidx = {}
        for img in self.img2set:
            set_dir = os.path.join(output, self.img2set[img])
            cat = self.img2cat[img]
            if cat not in catidx:
                catidx[cat] = -1

            catidx[cat] += 1

            if not is_dir(set_dir):
                os.makedirs(set_dir)

            cat_dir = os.path.join(set_dir, cat)
            if not is_dir(cat_dir):
                os.makedirs(cat_dir)

            if isinstance(img,Image):
                name, ext = img.name,img.ext
            else:
                name, ext = os.path.basename(img).rsplit(".", 1)
            suffix = ext if suffix is None else suffix.strip(".")

            new_name = "{}_{}.{}".format(cat, catidx[cat], suffix) if rename else "{}.{}".format(name, suffix)

            if isinstance(img,Image):
                imwrite(img.img,os.path.join(cat_dir, new_name))
            else:
                shutil.copy(img, os.path.join(cat_dir, new_name))

    @staticmethod
    def extract_by_dir(root_dir):
        img2cat = {}
        img2set = {}
        for s in os.listdir(root_dir):
            set_dir = os.path.join(root_dir, s)
            if is_dir(set_dir):
                for cat in os.listdir(set_dir):
                    cat_dir = os.path.join(set_dir, cat)
                    if is_dir(cat_dir):
                        for f in cat_dir:
                            img2cat[os.path.join(cat_dir, f)] = cat
                            img2set[os.path.join(cat_dir, f)] = s

        return img2cat, img2set

    @staticmethod
    def extract_by_samplefile(samplefile, split="\t"):
        samples = list_from_file(samplefile)
        _set = os.path.basename(samplefile).rsplit(".", 1)[0]

        img2cat = {}
        img2set = {}

        for s in samples:
            ss = s.split(split)
            img2cat[ss[0]] = ss[1]

            img2set[ss[0]] = _set

        return img2cat, img2set
