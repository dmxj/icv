# -*- coding: UTF-8 -*-
try:
    from .tf_obj import TfObjectDetector
except:
    pass

try:
    from .mb import MbDetector
except:
    pass

try:
    from .mmd import MmdetDetector
except:
    pass

from .service.client import DetectorClient