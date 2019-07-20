# -*- coding: UTF-8 -*-
from icv.image import imread,imshow,imwrite,imshow_bboxes
from .bbox_list import BBoxList
from icv.utils import EasyDict as edict
from .meta import SampleMeta,AnnoMeta
from .anno_shape import AnnoShape

class Anno(AnnoShape):
    def __init__(self,bbox=None,polys=None,mask=None,label=None,meta=None):
        super(Anno, self).__init__(bbox=bbox,polys=polys,mask=mask,label=label)
        self.meta = meta

class Sample(object):
    def __init__(self,name,image,annos=None,meta=None):
        for anno in annos:
            assert isinstance(anno,Anno)

        assert isinstance(meta,SampleMeta)

        self.name = name
        self.image = imread(image)
        self.meta = meta
        self.annos = [] if annos is None else annos

    @staticmethod
    def init(name, image, **kwargs):
        sample = Sample(name, image, meta=SampleMeta(**kwargs))
        return sample

    @property
    def count(self):
        return len(self.annos)

    @property
    def shape(self):
        return self.image.shape

    def statistics(self):
        sta = edict(count=self.count, cats=edict(), ratios=edict())
        for anno in self.annos:
            ratio = round(anno.bbox.ratio,1)
            if anno.label is not None:
                if anno.label not in sta["cats"]:
                    sta["cats"][anno.label] = 0
                sta["cats"][anno.label] += 1

                if anno.label not in sta["ratios"]:
                    sta["ratios"][anno.label] = edict()

                if ratio not in sta["ratios"][anno.label]:
                    sta["ratios"][anno.label][ratio] = 0

                sta["ratios"][anno.label][ratio] += 1

        return sta

    def vis(self,color="OrangeRed",with_bbox=True, is_show=False,save_path=None):
        image_drawed = self.image
        for anno in self.annos:
            if with_bbox:
                image_drawed = anno.bbox.draw_on_image(image_drawed,color=color)
            if anno.seg_model_mask:
                anno.mask.draw_on_image(image_drawed,color=color)
            if anno.seg_model_polys:
                anno.polys.draw_on_image(image_drawed,color=color)

        if is_show:
            imshow(image_drawed)

        if save_path is not None:
            imwrite(image_drawed,save_path)

        return image_drawed




class Sample2(object):
    def __init__(self,name,bbox_list,image):
        if not isinstance(bbox_list,BBoxList):
            bbox_list = BBoxList(bbox_list)
        self.name = name
        self.bbox_list = bbox_list
        self.image = imread(image)

        self.fields = edict()
        self.add_field("name",self.name)
        self.add_field("bbox_list",self.bbox_list)
        self.add_field("image",self.image)

    @staticmethod
    def init(name,bbox_list,image,**kwargs):
        sample = Sample(name,bbox_list,image)
        for key in kwargs:
            sample.add_field(key,kwargs[key])
        return sample

    @property
    def shape(self):
        return self.image.shape

    @property
    def count(self):
        return self.bbox_list.length

    def __getitem__(self, item):
        if self.has_field(item):
            return self.get_field(item)
        return None

    def __setitem__(self, key, value):
        self.add_field(key, value)

    def add_field(self, field, field_data):
        self.fields[field] = field_data

    def get_field(self, field):
        return self.fields[field]

    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    def vis(self,is_show=False,save_path=None):
        image_drawed = imshow_bboxes(self.image,self.bbox_list.bbox_list,is_show=is_show,save_path=save_path)
        return image_drawed

