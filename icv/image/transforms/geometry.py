# -*- coding: UTF-8 -*-
from __future__ import division

import cv2
import numpy as np
from ..io import imread
from .resize import imresize
from icv.utils.itis import is_np_array,is_seq,is_empty

def immerge(img_list, origin="x", resize=None):
    assert len(img_list) > 0
    assert origin in ["x", "y"], "param origin should be 'x' or 'y'"
    assert is_seq(img_list) and not is_empty(img_list), "param img_list should be a sequence"
    assert resize is None or (is_seq(resize) and len(resize) == 2)
    img_list = [imread(img) for img in img_list]
    if len(img_list) == 1:
        return img_list[0]

    if resize is None:
        if origin == "x":
            assert len(set([img.shape[0] for img in img_list])) == 1
        else:
            assert len(set([img.shape[1] for img in img_list])) == 1
    else:
        img_list = [imresize(img,resize) for img in img_list]

    merged_img_np = np.concatenate(img_list, axis=1 if origin == "x" else 0)
    return merged_img_np

def immix_up(img_list,resize=None,weight=None):
    assert len(img_list) > 0
    assert weight is None or (len(img_list) == len(weight) and sum(weight) == 1.0)
    img_list = [imread(img) for img in img_list]
    if len(img_list) == 1:
        return img_list[0]
    if resize is None:
        assert len(set([img.shape[:2] for img in img_list])) == 1
    else:
        img_list = [imresize(img, resize) for img in img_list]

    img_mixup = np.sum([img_list[i]*weight[i] for i in range(len(img_list))])
    return img_mixup

def imflip_lr(img):
    """Flip an image horizontally

    Args:
        img (ndarray): Image to be flipped.
    Returns:
        ndarray: The flipped image.
    """
    return np.flip(imread(img), axis=1)

def imflip_ud(img):
    """Flip an image vertically

    Args:
        img (ndarray): Image to be flipped.
    Returns:
        ndarray: The flipped image.
    """
    return np.flip(imread(img), axis=0)

def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple): Center of the rotation in the source image, by default
            it is the center of the image.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    img = imread(img)
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated

def imcrop(img, bboxes, scale=1.0, pad_fill=None):
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no padding.
        pad_fill (number or list): Value to be filled for padding, None for
            no padding.

    Returns:
        list or ndarray: The cropped image patches.
    """
    from icv.data.shape.transforms import bbox_clip, bbox_scaling

    img = imread(img)
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    if not is_np_array(bboxes):
        bboxes = np.array(bboxes)

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            if chn == 2:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start +
                  w, ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def impad(img, shape, pad_val=0):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or sequence): Values to be filled in padding areas.

    Returns:
        ndarray: The padded image.
    """
    img = imread(img)
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    pad = np.empty(shape, dtype=img.dtype)
    pad[...] = pad_val
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (number or sequence): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    img = imread(img)
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, (pad_h, pad_w), pad_val)
