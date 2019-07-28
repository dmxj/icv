# -*- coding: UTF-8 -*-
import os
import numpy as np

def is_str(input):
    return isinstance(input, str)


def is_list(input):
    return isinstance(input, list)


def is_seq(input):
    return isinstance(input, tuple) or isinstance(input, list)


def is_seq_equal(seq1, seq2, strict=False):
    """
    判断序列是否相等
    :param seq1: 序列1
    :param seq2: 序列2
    :param strict: 是否严格模式，即顺序也要匹配
    :return:
    """
    if not is_seq(seq1) or not is_seq(seq2) or len(seq1) != len(seq2):
        return False

    if strict:
        for i1, i2 in list(zip(seq1, seq2)):
            if i1 != i2:
                return False
        return True

    for i in seq1:
        if i not in seq2:
            return False
    return True


def is_np_array(input):
    return isinstance(input, np.ndarray)


def is_float_array(input):
    return is_np_array(input) and issubclass(input.dtype.type, np.floating)


def is_file(input):
    return is_str(input) and os.path.isfile(input) and os.path.exists(input)


def is_dir(input):
    return is_str(input) and os.path.isdir(input) and os.path.exists(input)


def is_path(input):
    return is_file(input) or is_dir(input)


def is_empty(input):
    if is_str(input):
        return input == ""
    if is_seq(input):
        return len(input) == 0
    if is_file(input):
        return float(os.path.getsize(input)) > 0
    if is_dir(input):
        from glob import glob
        return len(glob(os.path.join(input,"**","*"),recursive=True)) > 0

    return input is None


def is_valid_url(url):
    import re
    if re.match(r'^https?:/{2}\w.+$', url):
        return True
    return False


def is_py3():
    import sys
    if sys.version_info > (3, 0):
        return True
    return False


IS_PY3 = is_py3()
