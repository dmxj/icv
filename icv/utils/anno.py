# -*- coding: UTF-8 -*-
from lxml import etree
import os
import shutil

def load_voc_anno(anno_path,filter_empty=True):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    def _load_anno(xml):
        if not xml:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = _load_anno(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    with open(anno_path, 'r') as fid:
        xml_str = fid.read()
    if "<bndbox>" not in xml_str and filter_empty:
        return {}
    xml = etree.fromstring(xml_str)
    return _load_anno(xml)

def make_empty_voc_anno(**kwargs):
    empty_anno = {
        "folder":"",
        "filename":"",
        "source":{
            "database":"The VOC2012 Database",
            "annotation":"PASCAL VOC2007",
            "image":"flickr",
        },
        "size":{
            "width":"0",
            "height":"0",
            "depth":"0"
        },
        "segmented":"0",
        "object":{},
    }

    if "folder" in kwargs:
        empty_anno["folder"] = kwargs["folder"]

    if "filename" in kwargs:
        empty_anno["filename"] = kwargs["filename"]

    if "width" in kwargs:
        empty_anno["size"]["width"] = kwargs["width"]

    if "height" in kwargs:
        empty_anno["size"]["height"] = kwargs["height"]

    if "depth" in kwargs:
        empty_anno["size"]["depth"] = kwargs["depth"]

    return empty_anno

def reset_voc_dir(dist_dir,with_segment=True):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    dist_image_path = os.path.join(dist_dir, "JPEGImages")
    dist_anno_path = os.path.join(dist_dir, "Annotations")
    dist_imageset_path = os.path.join(dist_dir, "ImageSets", "Main")

    dist_segmentation_imageset_path = os.path.join(dist_dir, "Segmentation", "Main")
    dist_segmentation_class_image_path = os.path.join(dist_dir, "SegmentationClass")
    dist_segmentation_object_image_path = os.path.join(dist_dir, "SegmentationObject")

    os.makedirs(dist_image_path)
    os.makedirs(dist_anno_path)
    os.makedirs(dist_imageset_path)

    if with_segment:
        os.makedirs(dist_segmentation_imageset_path)
        os.makedirs(dist_segmentation_class_image_path)
        os.makedirs(dist_segmentation_object_image_path)

        return dist_image_path, dist_anno_path, dist_imageset_path, \
               dist_segmentation_imageset_path, dist_segmentation_class_image_path, dist_segmentation_object_image_path

    return dist_image_path, dist_anno_path, dist_imageset_path

def reset_coco_dir(dist_dir):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    shutil.rmtree(dist_dir)

    anno_dir = os.path.join(dist_dir,"annotations")
    train_image_dir = os.path.join(dist_dir,"train")
    test_image_dir = os.path.join(dist_dir,"test")
    val_image_dir = os.path.join(dist_dir,"val")

    os.makedirs(anno_dir)
    os.makedirs(train_image_dir)
    os.makedirs(test_image_dir)
    os.makedirs(val_image_dir)

    return anno_dir,train_image_dir,test_image_dir,val_image_dir
