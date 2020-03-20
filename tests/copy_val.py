#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
copy_val.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/12 下午8:00
"""

import pretrainedmodel#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
copy_val.py
Authors: rensike(rensike@baidu.com)
Date:    2019/9/12 下午8:00
"""

import pretrainedmodel

if __name__ == '__main__':
    anno = [
        {
            "name":1,
        },
        {
            "name":2,
        },
    ]

    for n in anno:
        n["name"] = 88

    print(anno)


if __name__ == '__main__':
    anno = [
        {
            "name":1,
        },
        {
            "name":2,
        },
    ]

    for n in anno:
        n["name"] = 88

    print(anno)
