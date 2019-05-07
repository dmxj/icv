# -*- coding: UTF-8 -*-
import os
from icv.core import errors
from PIL import Image
from icv.image import imread
from icv.utils import is_seq,is_empty
import cv2
import numpy as np

def imshow(img, win_name='', wait_time=0):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)

def immerge(img_list,origin="x",resize=False):
    assert origin in ["x","y"],"param origin should be 'x' or 'y'"
    assert is_seq(img_list) and not is_empty(img_list),"param img_list should be a sequence"
    if len(img_list) == 1:
        return imread(img_list[0])

    imgnp_list = []
    shape = ()
    for img in img_list:
        imgnp = imread(img)
        if not is_empty(shape):
            if resize:
                imgnp = cv2.resize(imgnp,shape)
            if (origin == "x" and imgnp.shape[0] != shape[0]) or (origin == "y" and imgnp.shape[1] != shape[1]):
                raise Exception("image shape must match exactly or use resize=True.")
        shape = (imgnp.shape[0], imgnp.shape[1])
        imgnp_list.append(imgnp)

    merged_img_np = np.concatenate(imgnp_list, axis=1 if origin == "x" else 0)
    return merged_img_np

def imshow_bboxes(img,bboxes,classes=None,scores=None):
    image = imread(img)
    return image
