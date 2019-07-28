# -*- coding: UTF-8 -*-
from icv.utils import EasyDict as edict
from icv.image import imread, imdraw_bbox
from ..shape.transforms import bbox_clip, bbox_scaling, bbox_extend
import numpy as np


class BBox(object):
    def __init__(self, xmin, ymin, xmax, ymax, label=None, **kwargs):
        assert xmin <= xmax, "xmax should be large than xmin."
        assert ymin <= ymax, "ymax should be large than ymin."

        self.set_bbox(xmin, ymin, xmax, ymax)

        self.fields = edict()
        for k in kwargs:
            self.add_field(k, kwargs[k])

        self.label = label

        self.add_field("xmin", self.xmin)
        self.add_field("ymin", self.ymin)
        self.add_field("xmax", self.xmax)
        self.add_field("ymax", self.ymax)
        self.add_field("center", self.center)
        self.add_field("width", self.width)
        self.add_field("height", self.height)

    @staticmethod
    def init_from(bbox, label=None):
        assert isinstance(bbox, BBox) or isinstance(bbox, list) or isinstance(bbox, np.ndarray)
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if isinstance(bbox, list):
            assert len(bbox) == 4
            bbox = BBox(xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3], label=label)
        return bbox

    def set_bbox(self, xmin, ymin, xmax, ymax):
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

        self._center = ((xmax + xmin) / 2, (ymax + ymin) / 2)
        self._width = xmax - xmin
        self._height = ymax - ymin

    def add_field(self, field, field_data):
        self.fields[field] = field_data

    def get_field(self, field):
        return self.fields[field]

    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    @property
    def bbox(self):
        return [self._xmin, self._ymin, self._xmax, self._ymax]

    @property
    def xmin(self):
        return self._xmin

    @property
    def ymin(self):
        return self._ymin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymax(self):
        return self._ymax

    @property
    def xmin_int(self):
        return int(np.round(self._xmin))

    @property
    def ymin_int(self):
        return int(np.round(self._ymin))

    @property
    def xmax_int(self):
        return int(np.round(self._xmax))

    @property
    def ymax_int(self):
        return int(np.round(self._ymax))

    @property
    def center(self):
        return self._center

    @property
    def center_x(self):
        return self.xmin + self._width / 2

    @property
    def center_y(self):
        return self.ymin + self._height / 2

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def ratio(self):
        return round(self._height / self._width, 2)

    @property
    def area(self):
        return self._height * self._width

    def iou(self, other):
        bbox = BBox.init_from(other)
        # TODO: check it
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.xmin, bbox.xmin)
        yA = max(self.ymin, bbox.ymin)
        xB = min(self.xmax, bbox.xmax)
        yB = min(self.ymax, bbox.ymax)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)
        boxBArea = (bbox.xmax - bbox.xmin + 1) * (bbox.ymax - bbox.ymin + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def intersection(self, other, default=None):
        bbox = BBox.init_from(other)
        xA = max(self.xmin, bbox.xmin)
        yA = max(self.ymin, bbox.ymin)
        xB = min(self.xmax, bbox.xmax)
        yB = min(self.ymax, bbox.ymax)

        if xA > xB or yA > yB:
            return default
        else:
            return BBox(xA, yA, xB, yB)

    def union(self, other):
        bbox = BBox.init_from(other)
        return BBox(
            xmin=min(self.xmin, bbox.xmin),
            ymin=min(self.ymin, bbox.ymin),
            xmax=max(self.xmax, bbox.xmax),
            ymax=max(self.ymax, bbox.ymax)
        )

    def is_fully_within_image(self, image):
        """
        Estimate whether the bounding box is fully inside the image area.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape
            and must contain at least two integers.
        Returns
        -------
        bool
            True if the bounding box is fully inside the image area. False otherwise.
        """
        shape = imread(image).shape
        height, width = shape[0:2]
        return self.xmin >= 0 and self.xmax < width and self.ymin >= 0 and self.ymax < height

    def is_partly_within_image(self, image):
        """
        Estimate whether the bounding box is at least partially inside the image area.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape
            and must contain at least two integers.
        Returns
        -------
        bool
            True if the bounding box is at least partially inside the image area. False otherwise.
        """
        shape = imread(image).shape
        height, width = shape[0:2]
        eps = np.finfo(np.float32).eps
        img_bb = BBox(xmin=0, xmax=width - eps, ymin=0, ymax=height - eps)
        return self.intersection(img_bb) is not None

    def is_out_of_image(self, image, fully=True, partly=False):
        """
        Estimate whether the bounding box is partially or fully outside of the image area.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use. If an ndarray, its shape will be used. If a tuple, it is
            assumed to represent the image shape and must contain at least two integers.
        fully : bool, optional
            Whether to return True if the bounding box is fully outside fo the image area.
        partly : bool, optional
            Whether to return True if the bounding box is at least partially outside fo the
            image area.
        Returns
        -------
        bool
            True if the bounding box is partially/fully outside of the image area, depending
            on defined parameters. False otherwise.
        """
        if self.is_fully_within_image(image):
            return False
        elif self.is_partly_within_image(image):
            return partly
        else:
            return fully

    def copy(self, xmin=None, ymin=None, xmax=None, ymax=None, label=None, **kwargs):
        """
        Create a shallow copy of the BoundingBox object.
        Parameters
        ----------
        xmin : None or number
            If not None, then the xmin coordinate of the copied object will be set to this value.
        ymin : None or number
            If not None, then the ymin coordinate of the copied object will be set to this value.
        xmax : None or number
            If not None, then the xmax coordinate of the copied object will be set to this value.
        ymax : None or number
            If not None, then the ymax coordinate of the copied object will be set to this value.
        label : None or string
            If not None, then the label of the copied object will be set to this value.
        Returns
        -------
        imgaug.BoundingBox
            Shallow copy.
        """
        return BBox(
            xmin=self.xmin if xmin is None else xmin,
            xmax=self.xmax if xmax is None else xmax,
            ymin=self.ymin if ymin is None else ymin,
            ymax=self.ymax if ymax is None else ymax,
            label=self.label if label is None else label,
            **kwargs
        )

    def deepcopy(self, xmin=None, ymin=None, xmax=None, ymax=None, label=None, **kwargs):
        """
        Create a deep copy of the BoundingBox object.
        Parameters
        ----------
        xmin : None or number
            If not None, then the xmin coordinate of the copied object will be set to this value.
        ymin : None or number
            If not None, then the ymin coordinate of the copied object will be set to this value.
        xmax : None or number
            If not None, then the xmax coordinate of the copied object will be set to this value.
        ymax : None or number
            If not None, then the ymax coordinate of the copied object will be set to this value.
        label : None or string
            If not None, then the label of the copied object will be set to this value.
        Returns
        -------
        imgaug.BoundingBox
            Deep copy.
        """
        return self.copy(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label, **kwargs)

    def draw_on_image(self, image, color=(0, 255, 0), thickness=1,
                      copy=True, raise_if_out_of_image=False):
        """
        Draw the bounding box on an image.
        Parameters
        ----------
        image : (H,W,C) ndarray(uint8)
            The image onto which to draw the bounding box.
        color : iterable of int, optional
            The color to use, corresponding to the channel layout of the image. Usually RGB.
        alpha : float, optional
            The transparency of the drawn bounding box, where 1.0 denotes no transparency and
            0.0 is invisible.
        size : int, optional
            The thickness of the bounding box in pixels. If the value is larger than 1, then
            additional pixels will be added around the bounding box (i.e. extension towards the
            outside).
        copy : bool, optional
            Whether to copy the input image or change it in-place.
        raise_if_out_of_image : bool, optional
            Whether to raise an error if the bounding box is fully outside of the
            image. If set to False, no error will be raised and only the parts inside the image
            will be drawn.
        Returns
        -------
        result : (H,W,C) ndarray(uint8)
            Image with bounding box drawn on it.
        """
        # from icv.image.vis import imdraw_bbox

        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception(
                "Cannot draw bounding box xmin=%.8f, ymin=%.8f, xmax=%.8f, ymax=%.8f on image with shape %s." % (
                    self.xmin, self.ymin, self.xmax, self.ymax, image.shape))

        result = np.copy(image) if copy else image

        display = "%s %.2f" % (
            self.label if self.label is not None else "",
            self.get_field("score") if self.has_field("score") else 0.0
        )

        return imdraw_bbox(
            result,
            xmin=self._xmin,
            ymin=self._ymin,
            xmax=self._xmax,
            ymax=self._ymax,
            color=color,
            thickness=thickness,
            display_str=display
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = self.__class__.__name__ + "("
        s += "xmin={}, ".format(self.xmin)
        s += "ymin={}, ".format(self.ymin)
        s += "xmax={}, ".format(self.xmax)
        s += "ymax={}, ".format(self.ymax)
        return s

    def clip(self, img_shape):
        clipped_bbox = bbox_clip(self, img_shape)[0]
        self.set_bbox(clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3])

    def scaling(self, scale, clip_shape=None):
        scaled_bbox = bbox_scaling(self, scale, clip_shape)[0]
        self.set_bbox(scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3])

    def extend(self, pad, clip_shape=None):
        extended_bbox = bbox_extend(self, pad, clip_shape)[0]
        self.set_bbox(extended_bbox[0], extended_bbox[1], extended_bbox[2], extended_bbox[3])
