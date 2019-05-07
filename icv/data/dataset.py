# -*- coding: UTF-8 -*-
import torch
import torch.utils.data


class IcvDataSet(torch.utils.data.Dataset):
    def __init__(self,ids=None,categories=None,keep_no_anno_image=True,one_index=False):
        self.ids = ids if ids else []
        self.categories = categories if categories else []
        self.keep_no_anno_image = keep_no_anno_image
        self.one_index = one_index
        self._build()

    def _build(self):
        if self.one_index:
            self.id2cat = {i+1: cat for i, cat in enumerate(self.categories)}
            self.cat2id = {cat: i+1 for i, cat in enumerate(self.categories)}
        else:
            self.id2cat = {i:cat for i, cat in enumerate(self.categories)}
            self.cat2id = {cat:i for i, cat in enumerate(self.categories)}

        self.id2index = {id:i for i,id in enumerate(self.ids)}
        self.index2id = {i:id for i,id in enumerate(self.ids)}

    def get_img_info(self,index):
        img_id = self.ids[index]
        return img_id

    @property
    def length(self):
        return len(self)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.get_sample(self.index2id[item])

    def reset_dir(self,dist_dir):
        pass

    def get_categories(self):
        pass

    def get_samples(self):
        samples = []
        for id in self.ids:
            sample = self.get_sample(id)
            if sample.bbox_list.length == 0 and not self.keep_no_anno_image:
                continue
            samples.append(sample)
        return samples

    def get_sample(self,id):
        pass

    def get_groundtruth(self,id):
        return self.get_sample(id)

    def vis(self,save_dir=None):
        pass