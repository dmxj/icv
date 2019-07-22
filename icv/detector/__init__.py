# -*- coding: UTF-8 -*-
from .service.client import DetectorClient

try:
    from .tf_obj import TfObjectDetector
except Exception as e:
    pass

try:
    from .mb import MbDetector
except Exception as e:
    pass

try:
    from .mmd import MmdetDetector
except Exception as e:
    pass

# __all__ = ['DetectorClient', 'TfObjectDetector', 'MbDetector', 'MmdetDetector']
