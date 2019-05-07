# -*- coding: UTF-8 -*-
import collections
import os

def is_str(input):
    return isinstance(input,str)

def is_list(input):
    return isinstance(input,list)

def is_seq(input):
    return isinstance(input,collections.Sequence)

def is_seq_equal(seq1,seq2,strict=False):
    if not is_seq(seq1) or not is_seq(seq2) or len(seq1) != len(seq2):
        return False

    if strict:
        for i1,i2 in list(zip(seq1,seq2)):
            if i1 != i2:
                return False
        return True

    for i in seq1:
        if i not in seq2:
            return False
    return True

def is_file(input):
    return os.path.isfile(input) and os.path.exists(input)

def is_empty(input):
    if is_str(input):
        return input == ""
    if is_seq(input):
        return len(input) == 0

    return input is None

def is_py3():
    import sys
    if sys.version_info > (3,0):
        return True
    return False

IS_PY3 = is_py3()
