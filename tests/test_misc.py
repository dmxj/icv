#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
test_misc.py
Authors: rensike(rensike@baidu.com)
Date:    2019/8/27 下午5:31
"""

from icv.utils import xml2json
import json

if __name__ == '__main__':
    data = xml2json("/Users/rensike/Work/icv/annotation/elf/segment/xml/outputs/000000009772.xml")
    print(data)
    # print(dict(data))
    # print(json.loads(json.dumps(data)))
