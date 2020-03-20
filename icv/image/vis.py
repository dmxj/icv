# -*- coding: UTF-8 -*-
from .io import imread, imwrite
from skimage import draw
from ..utils import is_seq, is_np_array
from .transforms import bgr2rgb
from ..vis.color import VIS_COLOR, get_color_tuple, get_text_color
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

TEXT_MARGIN = 2


def imshow(img):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
    """
    plt.axis('off')
    plt.imshow(imread(img,rgb_mode=True))
    plt.show()

def imshow_bboxes(
        img,
        bboxes,
        labels=None,
        scores=None,
        classes=None,
        score_thresh=0,
        masks=None,
        color=(244, 67, 54),
        thickness=1,
        use_normalized_coordinates=False,
        is_show=False,
        save_path=None):
    assert classes is None or is_seq(classes) or is_np_array(classes)
    assert labels is None or is_seq(labels) or is_np_array(labels)
    assert scores is None or is_seq(scores) or is_np_array(scores)
    assert masks is None or is_seq(masks) or is_np_array(masks)

    from icv.data.core import BBoxList, BBox

    if isinstance(bboxes, BBoxList):
        bboxes = np.array(bboxes.tolist())
    elif isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], BBox):
        bboxes = np.array([bbox.bbox for bbox in bboxes])
    else:
        bboxes = np.array(bboxes)

    image = imread(img)

    if bboxes.shape[0] == 0:
        if is_show:
            imshow(image)

        if save_path:
            imwrite(image, save_path)

        return image

    bboxes = bboxes[np.where(np.array(scores) >= score_thresh)] if scores is not None else bboxes

    if labels is not None:
        if not is_np_array(labels):
            labels = np.array(labels)
        labels = labels[np.where(np.array(scores) >= score_thresh)] if scores is not None else labels
        assert labels.shape[0] == bboxes.shape[0], "param labels's length is not equals to bboxes!"

    if masks is not None:
        if not is_np_array(masks):
            masks = np.array(masks)
        masks = masks[np.where(np.array(scores) >= score_thresh)] if scores is not None else masks
        assert masks.shape[0] == bboxes.shape[0], "param masks's length is not equals to bboxes!"

    if scores is not None:
        if not is_np_array(scores):
            scores = np.array(scores)
        scores = scores[np.where(scores >= score_thresh)]

        assert scores.shape[0] == bboxes.shape[0], "param scores's length is not equals to bboxes!"

    colorMap = {}
    default_color = color if color else "DarkOrange"
    if classes is not None:
        if is_np_array(classes):
            classes = list(classes)
        for cat in classes:
            colorMap[cat] = VIS_COLOR[classes.index(cat) % len(VIS_COLOR)]

    for ix, bbox in enumerate(bboxes):
        if isinstance(bbox, BBox):
            label = labels[ix] if labels is not None else bbox.label
            color = colorMap[label] if label in colorMap else default_color
            label = label if label is not None else ""
            label = label + ": " + str(round(scores[ix], 3)) if scores is not None else label
        else:
            label = labels[ix] if labels is not None else ""
            color = colorMap[label] if label in colorMap else default_color
            label = label + ": " + str(round(scores[ix], 3)) if scores is not None else label

        if masks is not None:
            image = imdraw_mask(image, masks[ix], color=color)

        if isinstance(bbox, BBox):
            image = imdraw_bbox(image, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, color, thickness, label,
                                use_normalized_coordinates)
        else:
            image = imdraw_bbox(image, bbox[0], bbox[1], bbox[2], bbox[3], color, thickness, label,
                                use_normalized_coordinates)

    if is_show:
        imshow(image)

    if save_path:
        imwrite(image, save_path)

    return image


def imdraw_bbox(image, xmin, ymin, xmax, ymax, color=(244, 67, 54), thickness=1, display_str="", text_color=None,
                use_normalized_coordinates=False):
    assert xmin <= xmax, "xmin shouldn't be langer than xmax!"
    assert ymin <= ymax, "ymin shouldn't be langer than ymax!"

    image = imread(image)
    image = image.copy()

    if xmin == xmax or ymin == ymax:
        return image

    im_height, im_width = image.shape[:2]
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    (left, right, top, bottom) = int(left), int(right), int(top), int(bottom)

    color = get_color_tuple(color)

    cv2.rectangle(image, (left, top), (right, bottom), color, thickness=thickness)
    if display_str == "":
        return image

    text_width, text_height, line_height = _get_text_size(display_str)
    text_left = left + TEXT_MARGIN
    text_top = top - TEXT_MARGIN

    if text_top < TEXT_MARGIN:
        text_top = bottom + TEXT_MARGIN

    if text_top + text_height + TEXT_MARGIN > im_height:
        text_top = top + TEXT_MARGIN

    text_color = get_color_tuple(text_color) if text_color is not None else get_text_color()
    image = imdraw_text(image, display_str, text_left, text_top, text_color=text_color, bg_color=color)
    return image


def imdraw_mask(image, mask, color='red', alpha=0.45):
    """Draws mask on an image.

    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: a uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
      ValueError: On incorrect data type for image or masks.
    """
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')

    image = imread(image)

    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                         'dimensions %s' % (image.shape[:2], mask.shape))

    color_mask = get_color_tuple(color)
    color_mask = np.array(color_mask, dtype=np.uint8)

    mask_bin = mask.astype(np.bool)
    image[mask_bin] = image[mask_bin] * alpha + color_mask * (1 - alpha)

    return image


def imdraw_polygons(image, polygons, color='red', alpha=0.45):
    image = imread(image)
    polygons = np.array(_format_polygons(polygons))
    polygons = np.squeeze(polygons)
    X, Y = polygons[:, 0], polygons[:, 1]
    rr, cc = draw.polygon(Y, X)

    rgb = get_color_tuple(color)
    draw.set_color(image, [rr, cc], color=rgb, alpha=alpha)

    return image


def _format_polygons(polygons):
    if is_np_array(polygons):
        polygons = polygons.tolist()
    ps = []
    for polygon in polygons:
        if is_seq(polygon[0]):
            ps.append([tuple(p) for p in polygon])
        else:
            if len(polygon) % 2 == 0:
                ps.append([(polygon[i], polygon[i + 1]) for i in range(len(polygon)) if i % 2 == 0])
    return ps


def _get_bbox_from_points(polygon):
    """
    根据polygon顶点获取bbox
    :param polygon:
    :return:
    """
    x_list = [p[0] for p in polygon]
    y_list = [p[1] for p in polygon]

    xmin = min(x_list)
    ymin = min(y_list)
    xmax = max(x_list)
    ymax = max(y_list)

    return xmin, ymin, xmax, ymax


def imdraw_polygons_with_bbox(
        image,
        polygons,
        label_list=None,
        is_show=False,
        save_path=None,
        alpha=0.45,
        outline=1,
        color_map=None,
        with_bbox=False,
        bbox_color=None,
        text_color=None
):
    image = imread(image)
    polygons = _format_polygons(polygons)

    if len(polygons) <= 0:
        if save_path is not None:
            imwrite(image, save_path)

        if is_show:
            imshow(image)
        return image

    assert label_list is None or len(polygons) == len(label_list)
    assert color_map is None or isinstance(color_map, dict)

    if isinstance(bbox_color, list):
        assert len(bbox_color) == len(polygons)
    if isinstance(text_color, list):
        assert len(text_color) == len(polygons)

    if bbox_color is None:
        bbox_color = random.choice(VIS_COLOR)

    if text_color is None:
        text_color = get_text_color()

    if label_list is None:
        label_list = [""] * len(polygons)

    tcolor_color_map = {}
    bcolor_color_map = {}
    color_list = []
    for ix, (polygon, label) in enumerate(list(zip(polygons, label_list))):
        tcolor = tcolor_color_map[label] if label in tcolor_color_map else None
        if tcolor is None:
            if isinstance(text_color, list):
                tcolor = text_color[ix]
            else:
                tcolor = text_color

        if label not in tcolor_color_map and tcolor is not None:
            tcolor_color_map[label] = tcolor

        bcolor = bcolor_color_map[label] if label in bcolor_color_map else None
        if bcolor is None:
            if isinstance(bbox_color, list):
                bcolor = bbox_color[ix]
            else:
                bcolor = bbox_color

        if label not in bcolor_color_map and bcolor is not None:
            bcolor_color_map[label] = bcolor

        points = [tuple(p) for p in polygon]
        xmin, ymin, xmax, ymax = _get_bbox_from_points(points)
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

        image = imdraw_polygons(image, polygon, color=bbox_color, alpha=alpha)

        if with_bbox:
            image = imdraw_bbox(image, left, top, right, bottom, color=bbox_color, thickness=outline, display_str=label,
                                text_color=tcolor)

        color_list.append(bcolor)

    if save_path is not None:
        imwrite(image, save_path)

    if is_show:
        imshow(image)

    return image


def _get_text_size(text, font_face=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.6, thickness=1):
    size = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width = size[0][0]
    text_height = size[0][1]
    return text_width, text_height, size[1]


def imdraw_text(image, text="-", x=0, y=0, font_scale=0.6, thickness=1, text_color=(255, 255, 255), with_bg=True,
                bg_color=(244, 67, 54), bg_alpha=60):
    image = imread(image)
    (height, width) = image.shape[:2]
    text_color = get_color_tuple(text_color)
    if with_bg:
        bg_color = get_color_tuple(bg_color, bg_alpha)
        text_width, text_height, line_height = _get_text_size(text, font_scale=font_scale, thickness=thickness)
        xmin = int(max(0, x - TEXT_MARGIN))
        ymin = int(max(0, y - TEXT_MARGIN - text_height))
        xmax = int(min(width, x + text_width + TEXT_MARGIN))
        ymax = int(min(height, y + TEXT_MARGIN))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bg_color, thickness=-1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return image
