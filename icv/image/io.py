# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from ..utils import is_str,is_seq,check_file_exist,USE_OPENCV2,mkdir,np_to_base64
import os
from PIL import Image

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

def imlist(img_dir, valid_exts=None, if_recursive=False):
    """
    List images under directory
    :param img_dir:
    :param valid_exts:
    :param if_recursive:
    :return:
    """
    from glob import glob
    if is_str(valid_exts):
        valid_exts = [valid_exts.strip(".")]
    valid_exts = list(valid_exts) if is_seq(valid_exts) else ["jpg","jpeg","bmp","tif","gif","png"]
    images = []
    for ext in valid_exts:
        images.extend(glob(os.path.join(img_dir,"**","*.%s" % ext),recursive=if_recursive))
    return images

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
    elif is_str(img_or_path):
        flag = imread_flags[flag] if is_str(flag) else flag
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        image = cv2.imread(img_or_path, flag)
        if flag != 0:
            image = image[...,::-1]
        return image
    else:
        try:
            return pil_img_to_np(img_or_path)
        except:
            raise TypeError('"img" must be a numpy array or a filename')

def imread_topil(img_or_path):
    if is_str(img_or_path):
        return Image.open(img_or_path)
    elif isinstance(img_or_path, np.ndarray):
        return np_img_to_pil(img_or_path)
    elif Image.isImageType(img_or_path):
        return img_or_path
    else:
        raise TypeError('"img" must be a numpy array or a filename or a PIL Image')

def imread_tob64(img_or_path):
    img = imread(img_or_path)
    img_b64 = np_to_base64(img)
    return img_b64.decode("utf-8")

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
    return np_img_to_pil(img).save(file_path)

    # return cv2.imwrite(file_path, img, params)

def pil_img_to_np(image):
    '''
    将图片加载为numpy数组
    :param image: PIL图片
    :return:
    '''
    (im_width, im_height) = image.size
    dim = 1 if len(np.array(image).shape) == 2 else np.array(image).shape[-1]
    return np.array(image.getdata()).reshape(
        (im_height, im_width, dim)).astype(np.uint8)


def np_img_to_pil(np_arr):
    '''
    将numpy array转换为图片
    :param np_arr:
    :return:
    '''
    im = Image.fromarray(np_arr.astype('uint8')).convert('RGB')
    return im

