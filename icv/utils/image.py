# -*- coding: utf-8 -* -
import numpy as np
from io import BytesIO
from PIL import Image
import base64

def base64_to_np(b64_code):
    image = Image.open(BytesIO(base64.b64decode(b64_code)))
    img = np.array(image)
    return img