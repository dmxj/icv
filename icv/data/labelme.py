# -*- coding: UTF-8 -*-
from .dataset import IcvDataSet
from icv.utils import is_seq
from icv.image.vis import imdraw_polygons_with_bbox,imshow_bboxes
from .core.bbox import BBox
from .core.bbox_list import BBoxList
from .core.sample import Sample
import os
import json

class LabelMe(IcvDataSet):
    def __init__(self,image_anno_path_list,keep_no_anno_image=True,categories=None,one_index=False):
        assert is_seq(image_anno_path_list)
        image_anno_path_list = list(image_anno_path_list)
        image_path_list,anno_path_list = list(zip(*image_anno_path_list))

        self.keep_no_anno_image = keep_no_anno_image

        self.ids = [_.rsplit(".",1)[0] for _ in image_path_list]
        self.id2imgpath = {id:image_path_list[ix] for ix,id in enumerate(self.ids)}
        self.id2annopath = {id:anno_path_list[ix] for ix,id in enumerate(self.ids)}

        self.get_samples()
        self.categories = categories if categories is not None else self.get_categories()

        super(LabelMe, self).__init__(self.ids, self.categories, self.keep_no_anno_image, one_index)

    def get_categories(self):
        categories = []
        for anno_sample in self.samples:
            label_list = anno_sample.bbox_list.labels
            if label_list:
                categories.extend(label_list)
        categories = list(set(categories))
        categories.sort()
        return categories

    def _get_bbox_from_points(self,points):
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

        return xmin,ymin,xmax,ymax

    def get_sample(self,id):
        """
        get sample
        :param id: image name
        :return:
        """
        anno_file = self.id2annopath[id]
        anno_data = json.load(open(anno_file,"r"))

        bbox_list = []
        if "shapes" in anno_data:
            shapes = anno_data["shapes"]
            for shape in shapes:
                if "points" in shape:
                    points = shape["points"]
                    xmin, ymin, xmax, ymax = self._get_bbox_from_points(points)
                    bbox_list.append(BBox(xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,label=shape["label"],shape=shape))

        sampleMeta = {k:anno_data[k] for k in anno_data if k != "shapes" and k != "imageData"}
        sample = Sample.init(
            name=id,
            bbox_list=BBoxList(bbox_list),
            image=self.id2imgpath[id],
            id=id,
            meta=sampleMeta,
        )
        sample.id = id
        # add other attr
        return sample

    def _write(self,anno_sample, dist_path):
        assert isinstance(anno_sample,Sample)
        anno_json = anno_sample.fields.meta
        anno_json["shapes"] = []
        for bbox in anno_sample.bbox_list:
            shape = dict(label=bbox.lable)
            for k in bbox.fields.shape:
                shape[k] = bbox.fields.shape[k]
            anno_json["shapes"].append(shape)
        json.dump(open(dist_path,"w"),anno_json)

    def vis(self, id=None, is_show=False, output_path=None):
        samples = []
        if id:
            sample = self.get_sample(id)
            samples.append(sample)
        else:
            samples = self.samples

        image_vis = []
        for sample in samples:
            image = sample.image
            polygons = []
            bboxes = []
            labels = []
            for bbox in sample.bbox_list:
                labels.append(bbox.lable)
                if bbox.fields.shape.shape_type == "polygon":
                    polygons.append(bbox.fields.shape.points)
                else:
                    bboxes.append(bbox)

            image = imdraw_polygons_with_bbox(image, polygons, labels, with_bbox=True,
                                              color_map=self.color_map)

            save_path = (output_path if id else os.path.join(output_path,sample.fields.meta.imagePath)) if output_path else None
            image = imshow_bboxes(image, BBoxList(bboxes), labels, is_show=is_show, save_path=save_path)

            image_vis.append(image)

        return image_vis[0] if id else image_vis

