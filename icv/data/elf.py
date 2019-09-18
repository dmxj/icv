# -*- coding: UTF-8 -*-
import os
import json
import random
import numpy as np
from skimage import measure
from tqdm import tqdm
import shutil
from .dataset import IcvDataSet
from ..data.core.bbox import BBox
from ..data.core.mask import Mask
from ..data.core.sample import Sample, Anno
from ..data.core.meta import AnnoMeta,SampleMeta
from ..vis.color import VIS_COLOR
from icv.image import imread
from icv.utils import is_file,is_seq,is_dict,xml2json,mkdir

class ElfFiles(object):
    def __init__(self,image_file,anno_file,attachment_dir=None):
        assert is_file(image_file)
        assert is_file(anno_file)
        assert attachment_dir is None or is_file(attachment_dir)

        self.image_file = image_file
        self.anno_file = anno_file
        self.attachment_file = attachment_dir

# TODO: bug 分割的label解析的有问题
class Elf(IcvDataSet):
    def __init__(self,image_anno_attachment_pathlist, split="trainval", keep_no_anno_image=True, categories=None,
                 one_index=False):
        assert is_seq(image_anno_attachment_pathlist)

        image_path_list, anno_path_list, attachment_dir_list = [],[], []
        for f in image_anno_attachment_pathlist:
            assert is_seq(f) or is_dict(f) or isinstance(f,ElfFiles)
            if is_seq(f):
                assert len(f) >= 2
                image_path_list.append(f[0])
                anno_path_list.append(f[1])

                if len(f) > 2:
                    attachment_dir_list.append(f[2])
                else:
                    attachment_dir_list.append(None)
            elif is_dict(f):
                assert "image_file" in f and "anno_file" in f
                image_path_list.append(f["image_file"])
                anno_path_list.append(f["anno_file"])

                if "attachment_dir" in f:
                    attachment_dir_list.append(f["attachment_dir"])
                else:
                    attachment_dir_list.append(None)
            else:
                image_path_list.append(f.image_file)
                anno_path_list.append(f.anno_file)
                attachment_dir_list.append(f.attachment_file)

        self.split = split
        self.keep_no_anno_image = keep_no_anno_image
        self.one_index = one_index

        self.ids = [os.path.basename(_).rsplit(".", 1)[0] for _ in image_path_list]
        self.id2imgpath = {id: image_path_list[ix] for ix, id in enumerate(self.ids)}
        self.id2annopath = {id: anno_path_list[ix] for ix, id in enumerate(self.ids)}
        self.id2attachpath = {id: attachment_dir_list[ix] for ix, id in enumerate(self.ids)}

        self.sample_db = {}
        self.color_map = {}
        self.get_samples()
        self.categories = categories if categories is not None else self.parse_categories()

        super(Elf, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index)
        print("there have %d samples in LabelMe dataset" % len(self.ids))
        print("there have %d categories in LabelMe dataset" % len(self.categories))

    def _get_bboxes_from_mask(self,mask_or_path):
        mask = imread(mask_or_path,0)
        contours = measure.find_contours(mask, 0.5)

        reshaped_contour = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            s = contour.ravel().tolist()
            assert len(s) % 2 == 0
            s = [(s[i], s[i + 1]) for i in range(len(s)) if i % 2 == 0]
            reshaped_contour.append(s)

        bboxes = []
        for i in range(len(reshaped_contour)):
            contour = np.array(reshaped_contour[i])
            xmin,xmax = int(np.min(contour[:,0])),int(np.max(contour[:,0]))
            ymin,ymax = int(np.min(contour[:,1])),int(np.max(contour[:,1]))
            bboxes.append([xmin,ymin,xmax,ymax])

        return bboxes

    def _parse_anno_item(self,ix,obj,id,anno_data):
        if "name" in obj:
            label = obj["name"]
            if label not in self.color_map:
                self.color_map[label] = random.choice(VIS_COLOR)
            if "bndbox" in obj:
                xmin = obj["bndbox"]["xmin"]
                ymin = obj["bndbox"]["ymin"]
                xmax = obj["bndbox"]["xmax"]
                ymax = obj["bndbox"]["ymax"]

                anno = Anno(
                    bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label),
                    label=label,
                    color=self.color_map[label],
                    meta=AnnoMeta()
                )

                return [anno]
            elif "attachments" in anno_data or ("doc" in anno_data and "attachments" in anno_data["doc"]): # for segmentation
                attachments = anno_data["attachments"] if "attachments" in anno_data else anno_data["doc"]["attachments"]
                def _parse_anno_from_attach(attach):
                    annos = []
                    if attach is not None and "file_name" in attach:
                        mask_path = os.path.join(self.id2attachpath[id],attach["file_name"])
                        img_mask = imread(mask_path,0)
                        bboxes = self._get_bboxes_from_mask(img_mask)
                        for bbox in bboxes:
                            in_others = False
                            for _bbox in bboxes:
                                if BBox.init_from(bbox).is_in(_bbox):
                                    in_others = True
                                    break
                            if in_others:
                                continue

                            xmin,ymin,xmax,ymax = bbox
                            anno = Anno(
                                bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label),
                                label=label,
                                color=self.color_map[label],
                                meta=AnnoMeta()
                            )

                            bin_mask = np.zeros_like(img_mask, np.uint8)
                            bin_mask[ymin:ymax, xmin:xmax] = img_mask[ymin:ymax, xmin:xmax]
                            bin_mask[np.where(bin_mask != 0)] = 1
                            anno.mask = Mask(bin_mask)

                            annos.append(anno)

                    return annos

                if is_seq(attachments):
                    return _parse_anno_from_attach(attachments[ix])
                elif is_dict(attachments) and "attachment" in attachments:
                    attachment = attachments["attachment"]
                    if is_seq(attachment):
                        return _parse_anno_from_attach(attachment[ix])
                    return _parse_anno_from_attach(attachment)

        return None

    def _parse_json_annotation(self,id,anno_data):
        annos = []
        if "labeled" in anno_data and not anno_data["labeled"]:
            return annos

        if "outputs" in anno_data and "object" in anno_data["outputs"]:
            for ix,obj in enumerate(anno_data["outputs"]["object"]):
                anno = self._parse_anno_item(ix,obj,id,anno_data)
                if anno is not None:
                    annos.extend(anno)

        return annos

    def _parse_xml_annotation(self,id,anno_data):
        annos = []
        if "labeled" in anno_data and not anno_data["labeled"]:
            return annos

        if "annotation" in anno_data:   # for pascal-voc (only detection)
            if "object" in anno_data["annotation"]:
                object = anno_data["annotation"]["object"]
                if not is_seq(object):
                    object = [object]
                for obj in object:
                    anno = self._parse_anno_item(0,obj,id,anno_data)
                    if anno is not None:
                        annos.extend(anno)
        elif "doc" in anno_data:    # for xml style (detection or segmentation)
            if "outputs" in anno_data["doc"] and "object" in anno_data["doc"]["outputs"] and "item" in anno_data["doc"]["outputs"]["object"]:
                items = anno_data["doc"]["outputs"]["object"]["item"]
                if not is_seq(items):
                    items = [items]
                for ix,obj in enumerate(items):
                    anno = self._parse_anno_item(ix,obj,id,anno_data)
                    if anno is not None:
                        annos.extend(anno)

        return annos

    def _load_annotation(self,id,anno_file):
        if str(anno_file).endswith(".json"):
            anno_data = json.load(open(anno_file, "r"))
            return self._parse_json_annotation(id,anno_data)
        elif str(anno_file).endswith(".xml"):
            anno_data = xml2json(anno_file)
            return self._parse_xml_annotation(id,anno_data)

    def get_sample(self, id):
        """
        get sample
        :param id: image name
        :return:
        """
        if id in self.sample_db:
            return self.sample_db[id]

        anno_file = self.id2annopath[id]
        annos = self._load_annotation(id,anno_file)
        img_file = self.id2imgpath[id]

        sample = Sample(
            name=id,
            image=img_file,
            annos=annos,
            meta=SampleMeta()
        )
        self.sample_db[id] = sample
        return sample

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