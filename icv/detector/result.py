# -*- coding: UTF-8 -*-
from icv.utils import is_seq
from icv.data import BBox,BBoxList
from icv.vis import imshow_bboxes
import numpy as np
import json

class DetectionResult(object):
    def __init__(self,det_bboxes,det_classes,det_scores,det_masks=None,det_time=0,det_image=None):
        assert is_seq(det_bboxes), "param det_bboxes should be a sequence."
        self._det_bboxes = BBoxList(np.array(det_bboxes).tolist())
        self._det_classes = np.array(det_classes)
        self._det_scores = np.array(det_scores)
        self._det_masks = det_masks
        self._det_time = det_time
        self._det_image = det_image

    @property
    def det_bboxes(self):
        return self._det_bboxes

    @property
    def det_classes(self):
        return self._det_classes

    @property
    def det_scores(self):
        return self._det_scores

    @property
    def det_masks(self):
        return self._det_masks

    @property
    def det_time(self):
        return self._det_time

    @property
    def det_image(self):
        return self._det_image

    @property
    def topk(self,k=1):
        if len(self) == 0:
            return (-1,0)

        assert k <= len(self),"param k should smaller than bbox count."

        topk_idx = np.argsort(self.det_scores)[-k:]
        topk_class = self.det_classes[topk_idx][::-1]
        topk_score = self.det_classes[topk_idx][::-1]

        _topk = list(zip(topk_class.tolist(),topk_score.tolist()))
        return _topk

    def __len__(self):
        return self.det_bboxes.shape[0]

    def to_dict(self):
        return dict(
            det_bboxes=self.det_bboxes,
            det_classes=self.det_classes,
            det_scores=self.det_scores,
            det_time=self.det_time,
            det_image=self.det_image
        )

    def to_json(self):
        return json.dumps(self.to_json())

    def vis(self,img):
        image_drawed = imshow_bboxes(img,self.det_bboxes,self.det_classes,self.det_scores)
        self._det_image = image_drawed
        return image_drawed
