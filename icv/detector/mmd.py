# -*- coding: UTF-8 -*-
try:
    import mmdet
except ModuleNotFoundError as e:
    raise Exception("You should install mmdetection first " \
                    "(https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) . " \
                    "Import mmdet Error!")

import os
import warnings
import numpy as np
from icv.data import BBox
from icv.utils import ckpt_load, Config, Timer, is_file, concat_list
from icv.image import imread, imwrite, imshow, imresize
from .detector import Detector
from .result import DetectionResult
import pycocotools.mask as maskUtils
from mmdet.core import get_classes
from mmdet.models import build_detector
from mmdet.apis import inference_detector


class MmdetDetector(Detector):
    def __init__(self, model_path, config_file, categories, iou_thr=0.5, score_thr=0.5, device=None):
        super(MmdetDetector, self).__init__(categories=categories, iou_thr=iou_thr, score_thr=score_thr, device=device)

        self.model_path = model_path
        self.config_file = config_file
        self._build_detector()

    def _build_detector(self):
        self.model = self._init_detector(self.config_file, self.model_path)

    def _init_detector(self, config, checkpoint=None, device='cuda:0'):
        """Initialize a detector from config file.

        Args:
            config (str or :obj:`mmcv.Config`): Config file path or the config
                object.
            checkpoint (str, optional): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        if isinstance(config, str):
            self.cfg = Config.fromfile(config)
        elif not isinstance(config, Config):
            raise TypeError('config must be a filename or Config object, '
                            'but got {}'.format(type(config)))
        self.cfg.model.pretrained = None
        model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        if checkpoint is not None:
            checkpoint = ckpt_load(model, checkpoint)
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['classes']
            else:
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use COCO classes by default.')
                model.CLASSES = get_classes('coco')
        model.cfg = self.cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model

    def _build_result(self, inference_result, inference_time=0, score_thr=-1):
        if isinstance(inference_result, tuple):
            bbox_result, segm_result = inference_result
        else:
            bbox_result, segm_result = inference_result, None

        detection_masks = []
        if segm_result is not None:
            segms = concat_list(segm_result)
            for segm in segms:
                mask = maskUtils.decode(segm)
                detection_masks.append(mask)

        detection_masks = np.array(detection_masks)

        preds = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

        detection_classes = np.concatenate(preds)
        bbox_result = np.vstack(bbox_result)

        detection_bboxes = []
        detection_scores = []

        for ix in range(bbox_result.shape[0]):
            detection_bboxes.append(
                BBox(bbox_result[ix, 0], bbox_result[ix, 1], bbox_result[ix, 2], bbox_result[ix, 3]))
            detection_scores.append(bbox_result[ix, 4])

        score_thr = score_thr if score_thr >= 0 else self.score_thr
        det_result = DetectionResult(
            det_bboxes=detection_bboxes,
            det_classes=detection_classes + 1,
            det_scores=detection_scores,
            det_masks=detection_masks,  # TODO: process segm_result
            det_time=inference_time,
            categories=self.categories,
            score_thr=score_thr
        )

        return det_result

    def inference(self, image, is_show=False, save_path=None, score_thr=-1):
        image_np = imread(image)
        timer = Timer()
        result = inference_detector(self.model, image_np, self.cfg)
        inference_time = timer.since_start()

        score_thr = score_thr if score_thr >= 0 else self.score_thr
        det_result = self._build_result(result, inference_time, score_thr)
        det_result.vis(image_np)

        if is_show:
            imshow(det_result.det_image)

        if save_path is not None:
            imwrite(det_result.det_image, save_path)

        return det_result

    def inference_batch(self, image_batch, save_dir=None, resize=None, score_thr=-1):
        image_np_list = [imread(img) for img in image_batch]
        if resize:
            image_np_list = [imresize(img_np, resize) for img_np in image_np_list]

        timer = Timer()
        results = inference_detector(self.model, image_np_list, self.cfg)

        inference_time = timer.since_start()
        score_thr = score_thr if score_thr >= 0 else self.score_thr

        det_result_list = []
        for ix, result in enumerate(results):
            det_result = self._build_result(result, inference_time, score_thr)
            det_result.vis(image_np_list[ix])

            if save_dir is not None:
                save_path = os.path.join(save_dir, os.path.basename(image_batch[ix])) if is_file(
                    image_batch[0]) else os.path.join(save_dir, str(ix) + ".jpg")
                imwrite(det_result.det_image, save_path)

            det_result_list.append(det_result)

        return det_result_list
