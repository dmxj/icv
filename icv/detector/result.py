# -*- coding: UTF-8 -*-
from ..utils import is_seq
from ..data.core import BBoxList
from ..image import imshow_bboxes
import numpy as np

class DetectionResult(object):
    def __init__(self,det_bboxes,det_classes,det_scores,det_masks=None,det_time=0,det_image=None,categories=None,score_thr=0):
        assert is_seq(det_bboxes), "param det_bboxes should be a sequence."
        self._score_thr = max(score_thr,0)
        self._det_scores = np.array(det_scores)

        self._det_bboxes = BBoxList(np.array(det_bboxes).tolist())
        self._det_classes = np.array(det_classes)
        self._det_masks = None if det_masks is None else np.array(det_masks)
        self._det_time = det_time
        self._det_image = det_image
        self.categories = categories
        self._det_labels = None
        if self.categories is not None:
            self._det_labels = [self.categories[_-1] for _ in self._det_classes]

        if self._score_thr >= 0:
            self.select_top_predictions()

    def select_top_predictions(self):
        filter_ids = np.where(np.array(self.det_scores) >= self._score_thr)
        self._det_scores = self._det_scores[filter_ids]
        self._det_bboxes.select(filter_ids[0])
        self._det_classes = self._det_classes[filter_ids]
        # TODO: 这里的过滤失效？？
        if self._det_masks is not None and self._det_masks.shape[0] > 0:
            self._det_masks = self._det_masks[filter_ids]

        if self.categories is not None:
            self._det_labels = [self.categories[_-1] for _ in self._det_classes]

    @property
    def det_bboxes(self):
        return self._det_bboxes

    @property
    def det_classes(self):
        return self._det_classes

    @property
    def det_labels(self):
        return self._det_labels

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

    def topk(self,k=1):
        if len(self) == 0:
            return [(-1,-1)]

        assert k <= len(self),"param k should smaller than bbox count."

        topk_idx = np.argsort(self.det_scores)[-k:]
        topk_class = self.det_classes[topk_idx][::-1]
        topk_score = self.det_scores[topk_idx][::-1]

        _topk = list(zip(topk_class.tolist(),topk_score.tolist()))
        return _topk

    @property
    def length(self):
        return len(self)

    def __len__(self):
        return len(self.det_bboxes)

    def to_dict(self):
        return dict(
            bboxes=self.det_bboxes.tolist(),
            classes=self.det_classes,
            scores=self.det_scores,
            time=self.det_time,
            image=self.det_image
        )

    # TODO: add image return
    def to_json(self):
        return dict(
            bboxes=self.det_bboxes.tolist(),
            classes=self.det_classes.tolist() if isinstance(self.det_classes,np.ndarray) else self.det_classes,
            scores=self.det_scores.tolist() if isinstance(self.det_classes,np.ndarray) else self.det_classes,
            time=self.det_time
        )

    def vis(self,img):
        image_drawed = imshow_bboxes(img,
                                     self.det_bboxes,
                                     classes=self.categories,
                                     labels=self.det_labels,
                                     scores=self.det_scores,
                                     masks=self.det_masks
                                     )
        self._det_image = image_drawed
        return image_drawed
