# -*- coding: utf-8 -* -
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import cv2
from .itis import is_np_array

def base64_to_np(b64_code):
    image = Image.open(BytesIO(base64.b64decode(b64_code)))
    img = np.array(image)
    return img

def np_to_base64(image_np):
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
    ims = [np.array(cv2.imread(image_file)) for image_file in image_files]
    shape = ims[0].shape

    if len(shape) < 3:
        (mean, stddv) = cv2.meanStdDev(np.array(ims))
        return mean[0],stddv[0]
    else:
        im_mean = []
        im_std = []
        for dim in range(shape[2]):
            (mean, stddv) = cv2.meanStdDev(np.array(ims)[:, :, :, dim])
            im_mean.append(mean[0][0])
            im_std.append(stddv[0][0])
        return im_mean,im_std