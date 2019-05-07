# -*- coding: UTF-8 -*-
import torch
from icv.utils import is_seq,is_file,labelmap_to_category_index

class Detector(object):
    def __init__(self,categories=None,labelmap_path=None,iou_thr=0.5,score_thr=0.5,device=None):
        assert is_seq(categories) or is_file(labelmap_path),"param categories and param labelmap_path should input at_least one."
        assert iou_thr > 0 and iou_thr < 1,"param iou_thr should > 0 and < 1."
        assert score_thr > 0 and score_thr < 1,"param score_thr should > 0 and < 1."

        self.categories = categories
        self.labelmap_path = labelmap_path
        self.iou_thr = iou_thr
        self.score_thr = score_thr

        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._labelmap()

    def _labelmap(self):
        if self.categories is not None:
            self.category_index = {id + 1: {"id": id + 1, "name": label} for id, label in enumerate(self.categories)}
            self.category_name_index = {self.category_index[cat_id]["name"]: self.category_index[cat_id] for cat_id in
                                        self.category_index}
        else:
            self.category_index = labelmap_to_category_index(self.labelmap_path,
                                                                                use_display_name=True)
            self.category_name_index = {self.category_index[cat_id]["name"]: self.category_index[cat_id] for cat_id in
                                        self.category_index}



    def inference(self,image,show=False,save_path=None):
        pass

    def inference_batch(self,image,force_resize=None):
        pass

    def start_server(self,port=8088):
        pass