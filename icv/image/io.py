# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import torch
from icv.utils import is_str,check_file_exist,USE_OPENCV2,mkdir
import os
from PIL import Image
import matplotlib.pyplot as plt

if not USE_OPENCV2:
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
else:
    from cv2 import CV_LOAD_IMAGE_COLOR as IMREAD_COLOR
    from cv2 import CV_LOAD_IMAGE_GRAYSCALE as IMREAD_GRAYSCALE
    from cv2 import CV_LOAD_IMAGE_UNCHANGED as IMREAD_UNCHANGED

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}

def imread(img_or_path, flag='color'):
    """Read an image.
    Args:
        img_or_path (ndarray or str or pillow Image): Either a numpy array or image path.
            If it is a numpy array (loaded image), then it will be returned
            as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path,torch.Tensor):
        return img_or_path.numpy()
    elif is_str(img_or_path):
        flag = imread_flags[flag] if is_str(flag) else flag
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        image = cv2.imread(img_or_path, flag)
        image = image[...,::-1]
        return image
    else:
        try:
            return load_image_into_numpy_array(img_or_path)
        except:
            raise TypeError('"img" must be a numpy array or a filename')

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        mkdir(dir_name)

    img = imread(img)
    return load_numpy_array_into_image(img).save(file_path)

    # return cv2.imwrite(file_path, img, params)

def load_image_into_numpy_array(image):
    '''
    将图片加载为numpy数组
    :param image: PIL图片
    :return:
    '''
    (im_width, im_height) = image.size
    dim = 1 if len(np.array(image).shape) == 2 else np.array(image).shape[-1]
    return np.array(image.getdata()).reshape(
        (im_height, im_width, dim)).astype(np.uint8)


def load_numpy_array_into_image(np_arr):
    '''
    将numpy array转换为图片
    :param np_arr:
    :return:
    '''
    im = Image.fromarray(np_arr.astype('uint8')).convert('RGB')
    return im

