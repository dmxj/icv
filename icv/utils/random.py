# -*- coding: UTF-8 -*-
import random
from .itis import is_seq

def random_sample(seq, ratio_or_num=None):
    assert is_seq(seq)
    seq = list(seq)

    if ratio_or_num is None:
        return random.choice(seq)

    count = ratio_or_num
    if ratio_or_num < 1:
        count = round(ratio_or_num * len(seq))
    count = min(int(count), len(seq))
    if count <= 0:
        return []
    return random.sample(seq, count)

def random_shuffle(seq):
    assert is_seq(seq)
    seq = list(seq)

    _seq = seq[::]
    random.shuffle(_seq)
    return _seq
