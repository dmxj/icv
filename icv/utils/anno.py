# -*- coding: UTF-8 -*-
from xml.etree import ElementTree as etree
from lxml.etree import Element, SubElement, ElementTree
import os
import shutil


def load_voc_anno(anno_path, filter_empty=False):
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


def save_voc_anno(anno_data, anno_dist_path):
    assert "folder" in anno_data
    assert "filename" in anno_data
    assert "size" in anno_data
    assert "width" in anno_data["size"]
    assert "height" in anno_data["size"]
    assert "depth" in anno_data["size"]
    assert "objects" in anno_data
    assert "segmented" in anno_data

    segmented = str(anno_data["segmented"])
    pose = "Unspecified"
    truncated = "0"
    difficult = "0"

    root = Element("annotation")
    SubElement(root, 'folder').text = anno_data["folder"]
    SubElement(root, 'filename').text = anno_data["filename"]

    source = SubElement(root, 'source')
    SubElement(source, 'database').text = "The VOC2007 Database"
    SubElement(source, 'annotation').text = "PASCAL VOC2007"
    SubElement(source, 'image').text = "flickr"

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(anno_data["size"]["width"])
    SubElement(size, 'height').text = str(anno_data["size"]["height"])
    SubElement(size, 'depth').text = str(anno_data["size"]["depth"])

    SubElement(root, 'segmented').text = segmented

    for object in anno_data["objects"]:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = object["name"]
        SubElement(obj, 'pose').text = str(object["pose"]) if "pose" in object else pose
        SubElement(obj, 'truncated').text = str(object["truncated"]) if "truncated" in object else truncated
        SubElement(obj, 'difficult').text = str(object["difficult"]) if "difficult" in object else difficult
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(object["bndbox"]["xmin"])
        SubElement(bndbox, 'ymin').text = str(object["bndbox"]["ymin"])
        SubElement(bndbox, 'xmax').text = str(object["bndbox"]["xmax"])
        SubElement(bndbox, 'ymax').text = str(object["bndbox"]["ymax"])

    tree = ElementTree(root)
    tree.write(anno_dist_path, encoding='utf-8', pretty_print=True)


def make_empty_coco_anno(**kwargs):
    empty_anno = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }

    for k in kwargs:
        empty_anno[k] = kwargs[k]

    return empty_anno


def make_empty_voc_anno(**kwargs):
    empty_anno = {
        "folder": "",
        "filename": "",
        "source": {
            "database": "The VOC2012 Database",
            "annotation": "PASCAL VOC2007",
            "image": "flickr",
        },
        "size": {
            "width": "0",
            "height": "0",
            "depth": "0"
        },
        "segmented": "0",
        "objects": {},
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


def reset_voc_dir(dist_dir, with_segment=True):
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

    anno_dir = os.path.join(dist_dir, "annotations")
    train_image_dir = os.path.join(dist_dir, "train")
    test_image_dir = os.path.join(dist_dir, "test")
    val_image_dir = os.path.join(dist_dir, "val")

    os.makedirs(anno_dir)
    os.makedirs(train_image_dir)
    os.makedirs(test_image_dir)
    os.makedirs(val_image_dir)

    return anno_dir, train_image_dir, test_image_dir, val_image_dir
