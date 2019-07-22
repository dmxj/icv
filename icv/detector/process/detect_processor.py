# -*- coding: utf-8 -* -
from .processor import Processor
from ...utils import base64_to_np

class DetectionProcessor(Processor):
    @classmethod
    def pre_process(self, inputs):
        return base64_to_np(inputs)

    @classmethod
    def post_process(self, outputs):
        if isinstance(outputs,list):
            outputs = [_.to_json() for _ in outputs]
        else:
            outputs = outputs.to_json()
        return outputs
