# -*- coding: utf-8 -* -
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from icv.utils.itis import is_np_array

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

