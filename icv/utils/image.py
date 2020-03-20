# -*- coding: utf-8 -* -
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import cv2
from .itis import is_np_array


def base64_to_np(b64_code):
    """
    base64 code convert to numpy array
    :param b64_code:
    :return:
    """
    image = Image.open(BytesIO(base64.b64decode(b64_code)))
    img = np.array(image)
    return img


def np_to_base64(image_np):
    """
    numpy array convert to base64 code
    :param image_np:
    :return:
    """
    assert is_np_array(image_np)
    output_buffer = BytesIO()
    img = Image.fromarray(image_np.astype('uint8')).convert('RGB')
    img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    base64_data = base64.b64encode(binary_data)
    return base64_data


def get_mean_std(image_files):
    """
    获取图片均值方差
    :param image_files:
    :return:
    """
    if len(image_files) <= 0:
        return [.0, .0, .0], [.0, .0, .0]
    means, stds = list(zip(*[cv2.meanStdDev(np.array(cv2.imread(image_file))) for image_file in image_files]))
    means = np.squeeze(np.array(means))
    stds = np.squeeze(np.array(stds))
    return np.mean(means, axis=0).tolist(), np.mean(stds, axis=0).tolist()


def list_images(img_root, recursive=True):
    """
    list image file path from directory
    :param img_root:
    :param recursive:
    :return:
    """
    import os
    from glob import glob

    imgs = []
    for ext in (
    '*.[gG][iI][fF]', '*.[pP][nN][gG]', '*.[jJ][pP][gG]', '*.[tT][iI][fF][fF]', '*.[jJ][pP][eE][gG]', '*.[bB][mM][pP]'):
        imgs.extend(glob(os.path.join(img_root, ext), recursive=recursive))
        if recursive:
            imgs.extend(glob(os.path.join(img_root, "**", ext), recursive=recursive))
    imgs = list(set(imgs))
    return imgs
