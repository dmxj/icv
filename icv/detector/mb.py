# -*- coding: UTF-8 -*-
try:
    import maskrcnn_benchmark
except ImportError as e:
    raise Exception("You should install maskrcnn_benchmark first " \
          "(https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) . " \
          "Import maskrcnn_benchmark Error!")

from .detector import Detector
from .result import DetectionResult
from icv.image import imwrite,imshow,imresize
import torch
from icv.image import imread
from icv.utils import Timer
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from icv.data import BBox
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg

class MbDetector(Detector):
    def __init__(self,model_path,config_file,categories,show_mask_heatmaps=False,iou_thr=0.5,score_thr=0.5,device=None):
        super(MbDetector, self).__init__(categories=categories, iou_thr=iou_thr, score_thr=score_thr,device=device)

        self.model_path = model_path
        self.config_file = config_file
        self.cfg = cfg.clone()
        self.cfg.merge_from_file(config_file)
        self.cpu_device = torch.device("cpu")

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.show_mask_heatmaps = show_mask_heatmaps

        self._build_detector()

    def _build_detector(self):
        self.model = build_detection_model(self.cfg)
        self.model.eval()
        self.model.to(self.device)

        checkpointer = DetectronCheckpointer(self.cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self._build_transform()

    def _build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def _compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def _compute_prediction_on_batch(self,original_image_list):
        """
        batch inference
        :param original_image_list: image list and each image as returned by OpenCV
        :return: batch inference result
        """
        image_list = [self.transforms(each_image) for each_image in original_image_list]
        image_list = to_image_list(image_list, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)

        def post_process(original_image,prediction):
            prediction.to(self.cpu_device)
            height, width = original_image.shape[:-1]
            prediction = prediction.resize((width, height))
            if prediction.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = prediction.get_field("mask")
                # always single image is passed at a time
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

            return prediction

        predictions = [post_process(original_image, pred) for original_image, pred in
                  list(zip(original_image_list, predictions))]
        return predictions

    def _build_result(self, inference_result, inference_time=0):
        assert isinstance(inference_result,BoxList),"inference result should be type of BoxList!"
        detection_classes = inference_result.get_field("labels")
        detection_scores = inference_result.get_field("scores")

        detection_bboxes = []
        bbox_result = inference_result.bbox
        for ix in enumerate(bbox_result.shape[0]):
            detection_bboxes.append(BBox(bbox_result[ix,0],bbox_result[ix,1],bbox_result[ix,2],bbox_result[ix,3]))

        segm_result = None
        if inference_result.has_field("mask"):
            segm_result = inference_result.get_field("mask")

        det_result = DetectionResult(
            det_bboxes=detection_bboxes,
            det_classes=detection_classes,
            det_scores=detection_scores,
            det_masks=segm_result,
            det_time=inference_time,
        )

        return det_result

    def inference(self, image, is_show=False, save_path=None):
        image_np = self.transforms(imread(image))
        timer = Timer()
        result = self._compute_prediction(image_np)
        inference_time = timer.since_start()

        det_result = self._build_result(result, inference_time)
        det_result.vis(image_np)

        if is_show:
            imshow(det_result.det_image)

        if save_path is not None:
            imwrite(det_result.det_image, save_path)

        return det_result

    def inference_batch(self, image_batch, resize=None):
        image_np_list = [imread(img) for img in image_batch]
        if resize:
            image_np_list = [imresize(img_np,resize) for img_np in image_np_list]

        if len(set([img_np.shape for img_np in image_np_list])) != 1:
            return [self.inference(img_np) for img_np in image_np_list]

        timer = Timer()
        predictions = self._compute_prediction_on_batch(image_np_list)
        inference_time = timer.since_start()

        det_result_list = []
        for ix,result in enumerate(predictions):
            det_result = self._build_result(result,inference_time)
            det_result.vis(image_np_list[ix])

            det_result_list.append(det_result)

        return det_result_list

    def start_server(self,port=8088,token=None):
        pass