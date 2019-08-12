# -*- coding: UTF-8 -*-
import json
from .itis import is_file

def encode_to_file(obj,filepath,overwrite=True):
    """
    save obj to json file, like json.dump()
    :param filepath:
    :param obj:
    :return:
    """
    if not overwrite and is_file(filepath):
        return
    return json.dump(obj,open(filepath,"w"))

def decode_from_file(filepath):
    """
    parse json file to obj, like json.load()
    :param filepath:
    :return:
    """
    return json.load(open(filepath))

def json_encode(obj):
    """
    encode obj to json string, like json.dumps()
    :param obj:
    :return:
    """
    return json.dumps(obj)


def json_decode(str, encoding="utf-8"):
    """
    decode json string to json object, like json.loads()
    :param str:
    :param encoding:
    :return:
    """
    return json.loads(str,encoding=encoding)
