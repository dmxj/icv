# -*- coding: UTF-8 -*-
import demjson

def encode_to_file(obj,filepath,encoding="utf-8", overwrite=False):
    """
    save obj to json file, like json.dump()
    :param filepath:
    :param obj:
    :param encoding:
    :return:
    """
    return demjson.encode_to_file(filepath,obj,encoding=encoding, overwrite=overwrite)

def decode_from_file(filepath,encoding="utf-8"):
    """
    parse json file to obj, like json.load()
    :param filepath:
    :param encoding:
    :return:
    """
    return demjson.decode_file(filepath,encoding=encoding)

def json_encode(obj, encoding="utf-8"):
    """
    encode obj to json string, like json.dumps()
    :param obj:
    :param encoding:
    :return:
    """
    return demjson.encode(obj, encoding=encoding)


def json_decode(str, encoding="utf-8"):
    """
    decode json string to json object, like json.loads()
    :param str:
    :param encoding:
    :return:
    """
    return demjson.decode(str, encoding=encoding)
