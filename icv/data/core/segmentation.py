# -*- coding: UTF-8 -*-
from icv.data.core.mask_list import MaskList
from icv.data.core.polygon_list import PolygonList

class SegmentationMask(object):

    """
    This class stores the segmentations for all objects in the image.
    It wraps BinaryMaskList and PolygonList conveniently.
    """

    def __init__(self, instances, size, mode="poly"):
        """
        Arguments:
            instances: two types
                (1) polygon
                (2) binary mask
            size: (width, height)
            mode: 'poly', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2

        assert isinstance(size[0], (int, float))
        assert isinstance(size[1], (int, float))

        if mode == "poly":
            self.instances = PolygonList(instances, size)
        elif mode == "mask":
            self.instances = MaskList(instances)
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        self.mode = mode
        self.size = tuple(size)

    # TODO: 尚未实现
    def transpose(self, method):
        flipped_instances = self.instances.transpose(method)
        return SegmentationMask(flipped_instances, self.size, self.mode)

    # TODO：尚未实现
    def crop(self, box):
        cropped_instances = self.instances.crop(box)
        cropped_size = cropped_instances.size
        return SegmentationMask(cropped_instances, cropped_size, self.mode)

    def resize(self, size, *args, **kwargs):
        resized_instances = self.instances.resize(size)
        resized_size = size
        return SegmentationMask(resized_instances, resized_size, self.mode)

    def to(self, *args, **kwargs):
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self

        if mode == "poly":
            converted_instances = self.instances.convert_to_polygon()
        elif mode == "mask":
            converted_instances = self.instances.convert_to_binarymask()
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        return SegmentationMask(converted_instances, self.size, mode)

    def get_mask_tensor(self):
        instances = self.instances
        if self.mode == "poly":
            instances = instances.convert_to_binarymask()
        # If there is only 1 instance
        return instances.masks.squeeze(0)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        selected_instances = self.instances.__getitem__(item)
        return SegmentationMask(selected_instances, self.size, self.mode)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_segmentation = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_segmentation
        raise StopIteration()

    next = __next__  # Python 2 compatibility

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.instances))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
