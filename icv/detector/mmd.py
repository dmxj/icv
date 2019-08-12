# -*- coding: UTF-8 -*-
try:
    import mmdet
except ModuleNotFoundError as e:
    raise Exception("You should install mmdetection first " \
                    "(https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) . " \
                    "Import mmdet Error!")

MMDET_VERSION = int(mmdet.__version__.split("+")[0].replace(".", ""))

import os
import numpy as np
from ..data.core import BBox
from ..utils import ckpt_load, Config, Timer, is_file, concat_list
from ..image import imread, imwrite, imshow, imresize
from .detector import Detector
from .result import DetectionResult
import torch
from mmdet.models import build_detector
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform


class MmdetDetector(Detector):
    def __init__(self, model_path, config_file, categories, iou_thr=0.5, score_thr=0.5, device=None):
        super(MmdetDetector, self).__init__(categories=categories, iou_thr=iou_thr, score_thr=score_thr, device=device)

        self.model_path = model_path
        self.config_file = config_file
        self._build_detector()
        self.img_transform = ImageTransform(size_divisor=self.cfg.data.test.size_divisor, **self.cfg.img_norm_cfg)

    def _build_detector(self):
        self.model = self._init_detector(self.config_file, self.model_path, device=self.device)

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
        try:
            if checkpoint is not None:
                checkpoint = ckpt_load(model, checkpoint)
            if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            elif self.categories is not None:
                model.CLASSES = self.categories
            else:
                model.CLASSES = ["unknown"]
            model.cfg = self.cfg
        except:
            pass

        model.to(device)
        model.eval()
        return model

    def _prepare_data(self, img):
        ori_shape = img.shape
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img,
            scale=self.cfg.data.test.img_scale,
            keep_ratio=self.cfg.data.test.get('resize_keep_ratio', True))
        img = to_tensor(img).to(self.device).unsqueeze(0)
        img_meta = [
            dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=False)
        ]
        return dict(img=[img], img_meta=[img_meta])

    def _inference_single(self, img):
        img = imread(img)
        data = self._prepare_data(img)
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        return result

    def _inference_batch(self, imgs):
        for img in imgs:
            yield self._inference_single(img)

    def _inference_detector(self, imgs):
        """Inference image(s) with the detector."""
        if not isinstance(imgs, list):
            return self._inference_single(imgs)
        else:
            return self._inference_batch(imgs)

    def _build_result(self, inference_result, inference_time=0, score_thr=-1):
        if isinstance(inference_result, tuple):
            bbox_result, segm_result = inference_result
        else:
            bbox_result, segm_result = inference_result, None

        detection_masks = []
        if segm_result is not None:
            import pycocotools.mask as maskUtils
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
            det_masks=detection_masks if detection_masks.shape[0] > 0 else None,  # TODO: process segm_result
            det_time=inference_time,
            categories=self.categories,
            score_thr=score_thr
        )

        return det_result

    def inference(self, image, is_show=False, save_path=None, score_thr=-1):
        image_np = imread(image)
        timer = Timer()
        result = self._inference_detector(image_np)
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
        results = self._inference_detector(image_np_list)

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
