#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
pp_det.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/26 上午11:26
"""

try:
    from paddle import fluid
    from ppdet.core.workspace import load_config, create
    from ppdet.modeling.model_input import create_feed
    from ppdet.data.data_feed import create_reader

    from ppdet.utils.eval_utils import parse_fetches
    from ppdet.utils.check import check_gpu
    import ppdet.utils.checkpoint as checkpoint
    from ppdet.utils.coco_eval import bbox2out, mask2out
except ModuleNotFoundError as e:
    raise Exception("You shoule install PaddleDetection first "
                    "(https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/PaddleDetection/docs/INSTALL.md). "
                    "Import PaddleDetection Error!")

import os
import numpy as np
from ..utils.itis import is_file, is_dir, is_seq
from ..utils.timer import Timer
from ..data.core import BBox
from ..image import imread, imwrite, imshow, imresize
from .detector import Detector
from .result import DetectionResult


class PPDetector(Detector):
    def __init__(self, model_dir, config_file, categories, iou_thr=0.5, score_thr=0, use_gpu=True):
        super(PPDetector, self).__init__(categories=categories, iou_thr=iou_thr, score_thr=score_thr)

        assert is_dir(model_dir), "model dir is not exist."
        assert is_file(config_file), "config file is not exist."
        assert is_seq(categories) and len(categories) > 0, "categories should be list"

        self.model_path = model_dir
        self.config_file = config_file
        self.categories = categories

        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.use_gpu = use_gpu

        self.init_cfg()
        self.init_test_feed()
        self.init_model()

    def init_cfg(self):
        self.cfg = load_config(self.config_file)
        if 'architecture' in self.cfg:
            self.main_arch = self.cfg.architecture
        else:
            raise ValueError("'architecture' not specified in config file.")

        self.cfg.use_gpu = self.use_gpu
        check_gpu(self.use_gpu)

        self.extra_keys = []
        if self.cfg['metric'] == 'COCO':
            self.extra_keys = ['im_info', 'im_id', 'im_shape']
        if self.cfg['metric'] == 'VOC':
            self.extra_keys = ['im_id', 'im_shape']

    def init_test_feed(self):
        if 'test_feed' not in self.cfg:
            self.test_feed = create(self.main_arch + 'TestFeed')
        else:
            self.test_feed = create(self.cfg.test_feed)

        self.clsid2catid = {i: i for i in range(len(self.categories))}

    def init_model(self):
        self.place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.model = create(self.main_arch)

        startup_prog = fluid.Program()
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                _, feed_vars = create_feed(self.test_feed, use_pyreader=False)
                self.test_fetches = self.model.test(feed_vars)
        self.infer_prog = infer_prog.clone(True)

        self.feeder = fluid.DataFeeder(place=self.place, feed_list=feed_vars.values())

        self.exe.run(startup_prog)
        if self.cfg.weights:
            checkpoint.load_checkpoint(self.exe, self.infer_prog, self.model_path)

        self.is_bbox_normalized = False
        if hasattr(self.model, 'is_bbox_normalized') and \
                callable(self.model.is_bbox_normalized):
            self.is_bbox_normalized = self.model.is_bbox_normalized()

    def _build_result(self, bbox_results, img_ids, segms_results=None, inference_time=0, score_thr=-1):
        det_results = []
        for img_id in img_ids:
            detection_bboxes = []
            detection_scores = []
            detection_classes = []

            for dt in np.array(bbox_results):
                if img_id != dt['image_id']:
                    continue
                catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']

                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h

                detection_bboxes.append(
                    BBox(xmin, ymin, xmax, ymax))
                detection_scores.append(score)
                detection_classes.append(catid)

            detection_masks = []
            if segms_results is not None:
                import pycocotools.mask as maskUtils
                for dt in np.array(segms_results):
                    if img_id != dt['image_id']:
                        continue

                    segm, score = dt['segmentation'], dt['score']
                    mask = maskUtils.decode(segm)
                    detection_masks.append(mask)

            detection_masks = np.array(detection_masks)
            score_thr = score_thr if score_thr >= 0 else self.score_thr
            det_result = DetectionResult(
                det_bboxes=detection_bboxes,
                det_classes=detection_classes,
                det_scores=detection_scores,
                det_masks=detection_masks if detection_masks.shape[0] > 0 else None,
                det_time=inference_time,
                categories=self.categories,
                score_thr=score_thr
            )
            det_results.append(det_result)

        return det_results

    def _infer(self, images, score_thr=-1):
        assert is_seq(images) and len(images) > 0
        self.test_feed.dataset.add_images(images)
        self.reader = create_reader(self.test_feed)

        keys, values, _ = parse_fetches(self.test_fetches, self.infer_prog, self.extra_keys)

        det_result_list = []

        for iter_id, data in enumerate(self.reader()):
            timer = Timer()
            outs = self.exe.run(self.infer_prog,
                                feed=self.feeder.feed(data),
                                fetch_list=values,
                                return_numpy=False)
            inference_time = timer.since_start()

            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }

            img_ids = res['im_id'][0]

            bbox_results = None
            segms_results = None
            if 'bbox' in res:
                bbox_results = bbox2out([res], self.clsid2catid, self.is_bbox_normalized)
            if 'mask' in res:
                segms_results = mask2out([res], self.clsid2catid,
                                         self.model.mask_head.resolution)

            det_result_list.append(self._build_result(bbox_results,
                                                      img_ids=img_ids,
                                                      segms_results=segms_results,
                                                      inference_time=inference_time,
                                                      score_thr=score_thr))

        return det_result_list

    def inference(self, image, is_show=False, save_path=None, score_thr=-1):
        image_np = imread(image)
        score_thr = score_thr if score_thr >= 0 else self.score_thr

        det_result = self._infer([image_np],score_thr=score_thr)[0][0]
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

        score_thr = score_thr if score_thr >= 0 else self.score_thr
        det_result_list = self._infer(image_np_list,score_thr=score_thr)[0]
        if save_dir is not None:
            for ix,det_result in enumerate(det_result_list):
                det_result.vis(image_np_list[ix])
                save_path = os.path.join(save_dir, os.path.basename(image_batch[ix])) if is_file(
                    image_batch[0]) else os.path.join(save_dir, str(ix) + ".jpg")
                imwrite(det_result.det_image, save_path)

        return det_result_list
