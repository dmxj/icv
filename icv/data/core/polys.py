# -*- coding: UTF-8 -*-
import numpy as np
from .kps import Keypoint
from icv.utils import EasyDict as edict
from icv.utils import is_np_array
from icv.image.vis import imdraw_polygons
import pycocotools.mask as mask_utils


class Polygon(object):
    def __init__(self, exterior, label=None, **kwargs):
        if isinstance(exterior, list):
            if not exterior:
                # for empty lists, make sure that the shape is (0, 2) and not (0,) as that is also expected when the
                # input is a numpy array
                self.exterior = np.zeros((0, 2), dtype=np.float32)
            elif isinstance(exterior[0], Keypoint):
                # list of Keypoint
                self.exterior = np.float32([[point.x, point.y] for point in exterior])
            elif not isinstance(exterior[0], (list, tuple)):
                assert len(exterior) % 2 == 0
                self.exterior = []
                for i, v in enumerate(exterior):
                    if i % 2 == 0:
                        self.exterior.append([v, exterior[i + 1]])
                self.exterior = np.float32(self.exterior)
            else:
                # list of tuples (x, y)
                self.exterior = np.float32([[point[0], point[1]] for point in exterior])
        else:
            assert is_np_array(exterior), ("Expected exterior to be a list of tuples (x, y) or "
                                           + "an (N, 2) array, got type %s") % (exterior,)

            assert exterior.ndim == 2 and exterior.shape[1] == 2, ("Expected exterior to be a list of tuples (x, y) or "
                                                                   "an (N, 2) array, got an array of shape %s") % (
                                                                      exterior.shape,)

            self.exterior = np.float32(exterior)

        if len(self.exterior) >= 2 and np.allclose(self.exterior[0, :], self.exterior[-1, :]):
            self.exterior = self.exterior[:-1]

        self.label = label

        self.bbox = None
        self.mask = None

        self.fields = edict()
        for k in kwargs:
            self.add_field(k, kwargs[k])

        self.label = label

        self.add_field("xx", self.xx)
        self.add_field("xx_int", self.xx_int)
        self.add_field("yy", self.yy)
        self.add_field("yy_int", self.yy_int)
        self.add_field("is_valid", self.is_valid)
        self.add_field("area", self.area)
        self.add_field("height", self.height)
        self.add_field("width", self.width)

    @staticmethod
    def init_from(polys, label=None):
        if isinstance(polys, list) or is_np_array(polys):
            return Polygon(polys, label)
        elif isinstance(polys, Polygon):
            return polys.deepcopy(label=label)
        else:
            raise TypeError("Type of polys should be list or ndarray or Polygon")

    def add_field(self, field, field_data):
        self.fields[field] = field_data

    def get_field(self, field):
        return self.fields[field]

    def has_field(self, field):
        return field in self.fields

    def fields(self):
        return list(self.fields.keys())

    @property
    def xx(self):
        """
        Return the x-coordinates of all points in the exterior.
        Returns
        -------
        (N,2) ndarray
            X-coordinates of all points in the exterior as a float32 ndarray.
        """
        return self.exterior[:, 0]

    @property
    def yy(self):
        """
        Return the y-coordinates of all points in the exterior.
        Returns
        -------
        (N,2) ndarray
            Y-coordinates of all points in the exterior as a float32 ndarray.
        """
        return self.exterior[:, 1]

    @property
    def xx_int(self):
        """
        Return the x-coordinates of all points in the exterior, rounded to the closest integer value.
        Returns
        -------
        (N,2) ndarray
            X-coordinates of all points in the exterior, rounded to the closest integer value.
            Result dtype is int32.
        """
        return np.int32(np.round(self.xx))

    @property
    def yy_int(self):
        """
        Return the y-coordinates of all points in the exterior, rounded to the closest integer value.
        Returns
        -------
        (N,2) ndarray
            Y-coordinates of all points in the exterior, rounded to the closest integer value.
            Result dtype is int32.
        """
        return np.int32(np.round(self.yy))

    @property
    def is_valid(self):
        """
        Estimate whether the polygon has a valid shape.
        To to be considered valid, the polygons must be made up of at least 3 points and have concave shape.
        Multiple consecutive points are allowed to have the same coordinates.
        Returns
        -------
        bool
            True if polygon has at least 3 points and is concave, otherwise False.
        """
        if len(self.exterior) < 3:
            return False
        return self.to_shapely_polygon().is_valid

    @property
    def area(self):
        """
        Estimate the area of the polygon.
        Returns
        -------
        number
            Area of the polygon.
        """
        if len(self.exterior) < 3:
            raise Exception("Cannot compute the polygon's area because it contains less than three points.")
        poly = self.to_shapely_polygon()
        return poly.area

    @property
    def height(self):
        """
        Estimate the height of the polygon.
        Returns
        -------
        number
            Height of the polygon.
        """
        yy = self.yy
        return max(yy) - min(yy)

    @property
    def width(self):
        """
        Estimate the width of the polygon.
        Returns
        -------
        number
            Width of the polygon.
        """
        xx = self.xx
        return max(xx) - min(xx)

    @property
    def polygons(self):
        return np.array(self.exterior).reshape(-1)

    def to_shapely_polygon(self):
        """
        Convert this polygon to a Shapely polygon.
        Returns
        -------
        shapely.geometry.Polygon
            The Shapely polygon matching this polygon's exterior.
        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry
        return shapely.geometry.Polygon([(point[0], point[1]) for point in self.exterior])

    def to_bounding_box(self):
        """
        Convert this polygon to a bounding box tightly containing the whole polygon.
        Returns
        -------
        imgaug.BoundingBox
            Tight bounding box around the polygon.
        """
        if self.bbox is not None:
            return self.bbox
        from .bbox import BBox

        xx = self.xx
        yy = self.yy
        self.bbox = BBox(xmin=min(xx), xmax=max(xx), ymin=min(yy), ymax=max(yy), label=self.label, **self.fields)
        return self.bbox

    def to_keypoints(self):
        """
        Convert this polygon's `exterior` to ``Keypoint`` instances.
        Returns
        -------
        list of imgaug.Keypoint
            Exterior vertices as ``Keypoint`` instances.
        """
        from .kps import Keypoint

        return [Keypoint(x=point[0], y=point[1]) for point in self.exterior]

    def to_mask(self, height, width):
        if self.mask is not None:
            return self.mask
        from .mask import Mask
        rles = mask_utils.frPyObjects([self.polygons], height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        self.mask = Mask(mask)
        return self.mask

    @staticmethod
    def from_shapely(polygon_shapely, label=None):
        """
        Create a polygon from a Shapely polygon.
        Note: This will remove any holes in the Shapely polygon.
        Parameters
        ----------
        polygon_shapely : shapely.geometry.Polygon
             The shapely polygon.
        label : None or str, optional
            The label of the new polygon.
        Returns
        -------
        imgaug.Polygon
            A polygon with the same exterior as the Shapely polygon.
        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        assert isinstance(polygon_shapely, shapely.geometry.Polygon)

        # polygon_shapely.exterior can be None if the polygon was instantiated without points
        if polygon_shapely.exterior is None or len(polygon_shapely.exterior.coords) == 0:
            return Polygon([], label=label)
        exterior = np.float32([[x, y] for (x, y) in polygon_shapely.exterior.coords])
        return Polygon(exterior, label=label)

    def copy(self, exterior=None, label=None, **kwargs):
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
        return self.deepcopy(exterior=exterior, label=label, **kwargs)

    def deepcopy(self, exterior=None, label=None, **kwargs):
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
        return Polygon(
            exterior=np.copy(self.exterior) if exterior is None else exterior,
            label=self.label if label is None else label,
            **kwargs
        )

    def draw_on_image(self,
                      image,
                      color=(0, 255, 0), alpha=0.6,
                      copy=True, raise_if_out_of_image=False):

        bbox = self.to_bounding_box()
        if raise_if_out_of_image and bbox.is_out_of_image(image):
            raise Exception(
                "Cannot draw mask xmin = %.8f, ymin = %.8f, xmax = %.8f, ymax = %.8f on image with shape %s." % (
                    bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, image.shape))

        result = np.copy(image) if copy else image

        result = imdraw_polygons(result, [self.exterior.tolist()], color=color, alpha=alpha)
        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        points_str = ", ".join(["(x=%.3f, y=%.3f)" % (point[0], point[1]) for point in self.exterior])
        return "Polygon([%s] (%d points), label=%s)" % (points_str, len(self.exterior), self.label)
