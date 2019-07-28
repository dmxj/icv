# -*- coding: UTF-8 -*-
import numpy as np
from skimage import measure
from icv.utils import EasyDict as edict
from icv.utils import is_seq, is_np_array,is_file
from .polys import Polygon
from icv.image.io import imread,imwrite
from icv.image.vis import imdraw_mask


class Mask(object):
    """
    Binary Mask
    """

    def __init__(self, mask, label=None, **kwargs):
        assert is_seq(mask) or is_np_array(mask)
        self.mask = np.array(mask).astype(np.uint8)
        assert len(self.mask.shape) == 2
        self.size = tuple(self.mask.shape)

        self.mask[self.mask != 0] = 1
        self.label = label

        self.polygon = None
        self.bbox = None

        self.fields = edict()
        for k in kwargs:
            self.add_field(k, kwargs[k])

        self.label = label
        if label is not None:
            self.add_field("label", self.label)

    @staticmethod
    def init_from(mask, label=None):
        if isinstance(mask, Mask):
            return mask.deepcopy(label=label)
        return Mask(mask, label=label)

    def add_field(self, field, field_data):
        self.fields[field] = field_data

    def get_field(self, field):
        return self.fields[field]

    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    def find_contours(self):
        contours = measure.find_contours(self.mask, 0.5)

        reshaped_contour = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            s = contour.ravel().tolist()
            assert len(s) % 2 == 0
            s = [(s[i], s[i + 1]) for i in range(len(s)) if i % 2 == 0]
            reshaped_contour.append(s)

        return reshaped_contour

        # mask = cv2.UMat(self.mask)
        # contour, hierarchy = cv2.findContours(
        #     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        # )
        #
        # reshaped_contour = []
        # for entity in contour:
        #     assert len(entity.shape) == 3
        #     assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
        #     reshaped_contour.append(entity.reshape(-1).tolist())
        #
        # return reshaped_contour

    def to_ploygons(self):
        if self.polygon is not None:
            return self.polygon
        contour = self.find_contours()
        if len(contour) > 0:
            self.polygon = Polygon.init_from(contour[0], self.label)
        else:
            self.polygon = Polygon.init_from([[0, 0]], self.label)
        return self.polygon

    def to_bounding_box(self):
        if self.bbox is not None:
            return self.bbox
        self.bbox = self.to_ploygons().to_bounding_box()
        return self.bbox

    def copy(self, mask=None, label=None, **kwargs):
        """
        Create a shallow copy of the Polygon object.
        Parameters
        ----------
        exterior : list of imgaug.Keypoint or list of tuple or (N,2) ndarray, optional
            List of points defining the polygon. See :func:`imgaug.Polygon.__init__` for details.
        label : None or str, optional
            If not None, then the label of the copied object will be set to this value.
        Returns
        -------
        imgaug.Polygon
            Shallow copy.
        """
        return self.deepcopy(mask=mask, label=label, **kwargs)

    def deepcopy(self, mask=None, label=None, **kwargs):
        """
        Create a deep copy of the Polygon object.
        Parameters
        ----------
        exterior : list of Keypoint or list of tuple or (N,2) ndarray, optional
            List of points defining the polygon. See `imgaug.Polygon.__init__` for details.
        label : None or str
            If not None, then the label of the copied object will be set to this value.
        Returns
        -------
        imgaug.Polygon
            Deep copy.
        """
        return Mask(
            mask=np.copy(self.mask) if mask is None else mask,
            label=self.label if label is None else label,
            **kwargs
        )

    def draw_on_image(self, image, color=(0, 255, 0), alpha=0.6,
                      copy=True, raise_if_out_of_image=False):
        assert image.shape[:2] == self.mask.shape

        bbox = self.to_bounding_box()
        if raise_if_out_of_image and bbox.is_out_of_image(image):
            raise Exception(
                "Cannot draw mask xmin = %.8f, ymin = %.8f, xmax = %.8f, ymax = %.8f on image with shape %s." % (
                    bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, image.shape))

        result = np.copy(image) if copy else image

        if isinstance(color, (tuple, list)):
            color = np.uint8(color)

        return imdraw_mask(image=result, mask=self.mask, color=color, alpha=alpha)

    def save(self, dist_path, id=1):
        m = self.mask[::]
        m[np.where(m != 0)] = id
        if is_file(dist_path):
            m0 = imread(dist_path,0)
            m = m + m0
            m[np.where(m>max(id,np.max(m0)))] = id

        imwrite(m, dist_path)
