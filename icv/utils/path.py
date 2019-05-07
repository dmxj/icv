# -*- coding: UTF-8 -*-
import shutil
import os
import six
from .itis import is_seq

def mkdir(dir_name,mode=0o777):
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)

def mkdir_force(dir_name, subdirs=None,mode=0o777):
    assert subdirs is None or is_seq(subdirs),"param subdirs should be none or a sequence"
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    mkdir(dir_name,mode)

    if subdirs is not None:
        for subdir in subdirs:
            mkdir(os.path.join(dir_name,subdir),mode)
