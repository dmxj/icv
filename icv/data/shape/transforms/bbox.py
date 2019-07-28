# -*- coding: UTF-8 -*-
import numpy as np
from icv.utils.itis import is_seq, is_np_array


def _transform_bboxes(bboxes):
    from icv.data.core import BBox, BBoxList
    if isinstance(bboxes, BBox):
        bboxes = BBoxList([bboxes])

    if isinstance(bboxes, BBoxList):
        bboxes = bboxes.to_np_array()

    if not is_np_array(bboxes):
        bboxes = np.array(bboxes)

    return bboxes


def bbox_clip(bboxes, img_shape):
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    bboxes = _transform_bboxes(bboxes)

    assert bboxes.shape[-1] % 4 == 0
    assert is_seq(img_shape) and len(img_shape) > 1
    clipped_bboxes = np.empty_like(bboxes, dtype=bboxes.dtype)
    clipped_bboxes[..., 0::2] = np.maximum(
        np.minimum(bboxes[..., 0::2], img_shape[1] - 1), 0)
    clipped_bboxes[..., 1::2] = np.maximum(
        np.minimum(bboxes[..., 1::2], img_shape[0] - 1), 0)
    return clipped_bboxes


def bbox_scaling(bboxes, scale, clip_shape=None):
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple, optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    bboxes = _transform_bboxes(bboxes)
    assert bboxes.shape[-1] % 4 == 0

    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def bbox_extend(bboxes, pad, clip_shape=None):
    """
    Extend bboxes to new position
    :param bboxes (ndarray): Shape(..., 4).
    :param pad: (int or 2 value list or 4 value list order from top、right、bottom、left)
    :param clip_shape(tuple, optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).
    :return:
        ndarray: Extended bboxes.
    """
    if isinstance(pad, int):
        t, r, b, l = (pad,) * 4
    elif is_seq(pad):
        if len(pad) == 1:
            t, r, b, l = (pad[0],) * 4
        elif len(pad) == 2:
            t, b = (pad[0],) * 2
            r, l = (pad[1],) * 2
        elif len(pad) >= 4:
            t, r, b, l = pad[0], pad[1], pad[2], pad[3]
        else:
            raise ValueError("pad should be one of [int(top|right|bottom|left),"
                             "tuple(2,top|bottom,right|left),tuple(4,top,right,bottom,left)]")
    else:
        raise ValueError("pad should be one of [int(top|right|bottom|left),"
                         "tuple(2,top|bottom,right|left),tuple(4,top,right,bottom,left)]")

    bboxes = _transform_bboxes(bboxes)
    assert bboxes.shape[-1] % 4 == 0

    extended_bboxes = bboxes.copy()
    extended_bboxes[..., 0] -= l
    extended_bboxes[..., 1] -= t
    extended_bboxes[..., 2] += r
    extended_bboxes[..., 3] += b

    if clip_shape is not None:
        return bbox_clip(extended_bboxes, clip_shape)
    else:
        return extended_bboxes
