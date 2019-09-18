# -*- coding: UTF-8 -*-
import os
import json
import itertools
from .itis import is_dict,is_seq
from xml.etree import ElementTree as etree
from xmljson import badgerfish as bf

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))

def xml2json(xml_path_or_content):
    if str(xml_path_or_content).endswith(".xml"):
        with open(xml_path_or_content, 'r') as fid:
            xml_str = fid.read()
    else:
        xml_str = xml_path_or_content
    bf_data = bf.data(etree.fromstring(xml_str))
    bf_data = json.loads(json.dumps(bf_data))

    def _format(data):
        if is_dict(data):
            for k in data:
                if k == '$':
                    return data[k]
                data[k] = _format(data[k])
        elif is_seq(data):
            for i in range(len(data)):
                data[i] = _format(data[i])
        return data

    bf_data = _format(bf_data)
    return bf_data

