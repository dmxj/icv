#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
conf_dict_debug.py
Authors: rensike(rensike@baidu.com)
Date:    2019/8/15 下午9:11
"""

conf_count_kuang = {str(i / 100): {"a":0,"b":0} for i in list(range(0, 100, 1))}
bboxes = [[1,2,3,4,0.74]]

if __name__ == '__main__':
    for i in range(len(bboxes)):
        for key in conf_count_kuang.keys():
            if bboxes[i][4] > float(key):
                print('{} +1'.format(key))
                class_name = "a"
                conf_count_kuang[key][class_name] += 1
        print('[DEBUG]: {}'.format(conf_count_kuang))
        exit(0)


