# -*- coding: UTF-8 -*-
import os
from ..image import imread, imwrite, imresize, imshow
from .detector import Detector
from ..data.core import BBox
from ..utils import Timer, is_file, is_seq
from .result import DetectionResult
try:
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print("you are not install tensorflow or numpy")
from .utils import ops as utils_ops

class TfObjectDetector(Detector):
    def __init__(self, model_path, labelmap_path=None, categories=[], iou_thr=0.5, score_thr=0.5, device=None):
        assert is_file(model_path)
        assert (is_seq(categories) and len(categories) > 0) or is_file(labelmap_path)
        super(TfObjectDetector, self).__init__(categories=categories, labelmap_path=labelmap_path, iou_thr=iou_thr,
                                               score_thr=score_thr, device=device)

        self.model_path = model_path
        self._build_detector()

    def _build_detector(self):
        self._load_model()
        self._load_sess()

    def _load_model(self):
        '''
        加载模型
        :return:
        '''
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph

    def _load_sess(self):
        '''
        初始化session
        :return:
        '''
        with self.graph.as_default():
            self.sess = tf.Session()
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    def _reframe_detection_mask(self, image_np):
        '''
        对图片的分割掩码进行转换
        :param image:
        :return:
        '''
        if 'detection_masks' in self.tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            self.tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)

    def inference(self, image, is_show=False, save_path=None, score_thr=-1):
        image_np = imread(image)
        img_height, img_width = image_np.shape[:2]
        self._reframe_detection_mask(image_np)
        timer = Timer()
        # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: np.expand_dims(image_np, 0)})
        inference_time = timer.since_start()

        # all outputs are float32 numpy arrays, so convert types as appropriate
        num_detections = int(output_dict['num_detections'][0])
        detection_classes = output_dict[
            'detection_classes'][0].astype(np.uint8)
        # ymin,xmin,ymax,xmax
        detection_boxes = output_dict['detection_boxes'][0]
        detection_scores = output_dict['detection_scores'][0]
        detection_masks = None
        if 'detection_masks' in output_dict:
            detection_masks = output_dict['detection_masks'][0]

        detection_bboxes = [
            BBox(
                xmin=det_box[1] * img_width,
                ymin=det_box[0] * img_height,
                xmax=det_box[3] * img_width,
                ymax=det_box[2] * img_height
            )
            for det_box in detection_boxes
        ]

        score_thr = score_thr if score_thr >= 0 else self.score_thr
        det_result = DetectionResult(
            det_bboxes=detection_bboxes,
            det_classes=detection_classes,
            det_scores=detection_scores,
            det_masks=detection_masks,
            det_time=inference_time,
            categories=self.categories,
            score_thr=score_thr
        )

        det_result.vis(image_np)

        if is_show:
            imshow(det_result.det_image)

        if save_path is not None:
            imwrite(det_result.det_image, save_path)

        return det_result

    def inference_batch(self, image_batch, save_dir=None, resize=None, score_thr=-1):
        self._reframe_detection_mask(imread(image_batch[0]))
        if isinstance(image_batch, np.ndarray):
            image_np_batch = image_batch
        else:
            image_np_list = [imread(img) for img in image_batch]
            if len(set([img_np.shape for img_np in image_np_list])) != 1:
                if resize is None:
                    # inference one by one
                    return [self.inference(img_np) for img_np in image_np_list]
                    # raise Exception("if input image list not have same size, param resize must be a tuple of (width,height).")
            if resize is None:
                image_np_batch = np.array(image_np_list)
            else:
                image_np_batch = np.array(
                    [imresize(img_np, resize) for img_np in image_np_list]
                )

        # Run inference
        timer = Timer()
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: image_np_batch})

        inference_time = timer.since_start()

        score_thr = score_thr if score_thr >= 0 else self.score_thr
        det_result_list = []
        for ix, image in enumerate(image_np_batch):
            img_height, img_width = image.shape[:2]
            # all outputs are float32 numpy arrays, so convert types as appropriate
            num_detections = int(output_dict['num_detections'][ix])
            detection_classes = output_dict[
                'detection_classes'][ix].astype(np.uint8)
            detection_boxes = output_dict['detection_boxes'][ix]
            detection_scores = output_dict['detection_scores'][ix]
            detection_masks = None
            if 'detection_masks' in output_dict:
                detection_masks = output_dict['detection_masks'][ix]

            detection_bboxes = [
                BBox(
                    xmin=det_box[1] * img_width,
                    ymin=det_box[0] * img_height,
                    xmax=det_box[3] * img_width,
                    ymax=det_box[2] * img_height
                )
                for det_box in detection_boxes
            ]

            det_result = DetectionResult(
                det_bboxes=detection_bboxes,
                det_classes=detection_classes,
                det_scores=detection_scores,
                det_masks=detection_masks,
                det_time=inference_time,
                score_thr=score_thr
            )
            det_result.vis(image)

            if save_dir is not None:
                save_path = os.path.join(save_dir, os.path.basename(image_batch[ix])) if is_file(
                    image_batch[0]) else os.path.join(save_dir, str(ix) + ".jpg")
                imwrite(det_result.det_image, save_path)

            det_result_list.append(det_result)

        return det_result_list
