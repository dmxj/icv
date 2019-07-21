# -*- coding: UTF-8 -*-
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
# from icv.data import BBoxList, BBox
from icv.image import imread, imread_topil, imwrite,pil_img_to_np
from icv.utils import is_seq, is_empty, is_np_array,is_str
from icv.vis.color import STANDARD_COLORS, MASK_COLORS
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, win_name='', wait_time=0):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    plt.axis('off')
    plt.imshow(imread(img))
    plt.show()
    # cv2.imshow(win_name, imread(img))
    # cv2.waitKey(wait_time)


def immerge(img_list, origin="x", resize=False):
    assert origin in ["x", "y"], "param origin should be 'x' or 'y'"
    assert is_seq(img_list) and not is_empty(img_list), "param img_list should be a sequence"
    if len(img_list) == 1:
        return imread(img_list[0])

    imgnp_list = []
    shape = ()
    for img in img_list:
        imgnp = imread(img)
        if not is_empty(shape):
            if resize:
                imgnp = cv2.resize(imgnp, shape)
            if (origin == "x" and imgnp.shape[0] != shape[0]) or (origin == "y" and imgnp.shape[1] != shape[1]):
                raise Exception("image shape must match exactly or use resize=True.")
        shape = (imgnp.shape[0], imgnp.shape[1])
        print("===> image shape:", imgnp.shape)
        imgnp_list.append(imgnp)

    merged_img_np = np.concatenate(imgnp_list, axis=1 if origin == "x" else 0)
    print("===> merged image shape:", merged_img_np.shape)
    return merged_img_np


def imshow_bboxes(
        img,
        bboxes,
        labels=None,
        scores=None,
        classes=None,
        score_thresh=0,
        masks=None,
        color="red",
        thickness=1,
        use_normalized_coordinates=False,
        is_show=False,
        save_path=None):
    assert classes is None or is_seq(classes) or isinstance(classes, np.ndarray)
    assert labels is None or is_seq(labels) or isinstance(labels, np.ndarray)
    assert scores is None or is_seq(scores) or isinstance(scores, np.ndarray)
    assert masks is None or is_seq(masks) or isinstance(masks, np.ndarray)

    from icv.data import BBoxList, BBox

    if isinstance(bboxes, BBoxList):
        bboxes = np.array(bboxes.tolist())
    elif isinstance(bboxes, list) and isinstance(bboxes[0], BBox):
        bboxes = np.array([bbox.bbox for bbox in bboxes])
    else:
        bboxes = np.array(bboxes)

    image = imread(img)
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    if bboxes.shape[0] == 0:
        if is_show:
            imshow(image)

        if save_path:
            imwrite(image, save_path)

        return image

    bboxes = bboxes[np.where(scores >= score_thresh)] if scores is not None else bboxes

    if labels is not None:
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        labels = labels[np.where(scores >= score_thresh)] if scores is not None else labels
        assert labels.shape[0] == bboxes.shape[0], "param labels's length is not equals to bboxes!"

    if masks is not None:
        if not isinstance(masks, np.ndarray):
            masks = np.array(masks)
        masks = masks[np.where(scores >= score_thresh)] if scores is not None else masks
        assert masks.shape[0] == bboxes.shape[0], "param masks's length is not equals to bboxes!"

    if scores is not None:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        scores = scores[np.where(scores >= score_thresh)]

        assert scores.shape[0] == bboxes.shape[0], "param scores's length is not equals to bboxes!"

    colorMap = {}
    default_color = color if color else "DarkOrange"
    if classes is not None:
        if isinstance(classes, np.ndarray):
            classes = list(classes)
        for cat in classes:
            colorMap[cat] = STANDARD_COLORS[classes.index(cat) % len(STANDARD_COLORS)]

    for ix, bbox in enumerate(bboxes):
        if isinstance(bbox, BBox):
            label = labels[ix] if labels is not None else bbox.lable
            color = colorMap[label] if label in colorMap else default_color
            label = label if label is not None else ""
            label = label + ": " + str(round(scores[ix], 3)) if scores is not None else label
        else:
            label = labels[ix] if labels is not None else ""
            color = colorMap[label] if label in colorMap else default_color
            label = label + ": " + str(round(scores[ix], 3)) if scores is not None else label

        if masks is not None:
            image_pil = imdraw_mask(image_pil, masks[ix], color=color)

        if isinstance(bbox, BBox):
            image_pil = imdraw_bbox(image_pil, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, color, thickness, label,
                                    use_normalized_coordinates)
        else:
            image_pil = imdraw_bbox(image_pil, bbox[0], bbox[1], bbox[2], bbox[3], color, thickness, label,
                                    use_normalized_coordinates)

    np.copyto(image, np.array(image_pil))

    if is_show:
        imshow(image)

    if save_path:
        imwrite(image, save_path)

    return image


def imdraw_bbox(image, xmin, ymin, xmax, ymax, color="red", thickness=1, display_str="",
                use_normalized_coordinates=False):
    assert xmin <= xmax, "xmin shouldn't be langer than xmax!"
    assert ymin <= ymax, "ymin shouldn't be langer than ymax!"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')

    if xmin == xmax or ymin == ymax:
        return image

    draw = ImageDraw.Draw(image)
    im_height, im_width = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    if display_str == "":
        return image

    rgb = ImageColor.getrgb(color) if is_str(color) else color
    rgba = rgb + (128,)

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    display_str_heights = font.getsize(display_str)[1]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * display_str_heights

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    if text_width == 0:
        text_width = len(display_str) * 7
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=rgba)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)

    return pil_img_to_np(image)

def imdraw_mask(image, mask, color='red', alpha=0.4):
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

    color_mask = ImageColor.getrgb(color) if is_str(color) else color
    color_mask = np.array(color_mask, dtype=np.uint8)

    mask_bin = mask.astype(np.bool)
    image[mask_bin] = image[mask_bin] * alpha + color_mask * (1 - alpha)

    return image


def imdraw_polygons(image, polygons, color='red', alpha=0.4):
    image = imread(image)

    polygons = _format_polygons(polygons)

    rgb = ImageColor.getrgb(color) if is_str(color) else color
    alpha = int(alpha * 255) if alpha <= 1 else int(alpha)
    rgba = rgb + (alpha,)
    pil_image = Image.fromarray(image)

    draw = ImageDraw.Draw(pil_image)
    for polygon in polygons:
        points = [tuple(p) for p in polygon]
        draw.polygon(points, fill=rgba)

    np.copyto(image, np.array(pil_image.convert('RGB')))
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
        name_list=None,
        is_show=False,
        save_path=None,
        alpha=0.5,
        outline=1,
        color_map=None,
        with_bbox=False,
        bbox_color='black',
        text_color='black'
):
    image = imread(image)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, mode="RGBA")

    polygons = _format_polygons(polygons)

    if len(polygons) <= 0:
        if save_path is not None:
            pil_image.save(save_path)

        if is_show:
            imshow(image)
        return image

    assert name_list is None or len(polygons) == len(name_list)
    assert color_map is None or isinstance(color_map, dict)

    color_map_history = {}
    for ix, polygon in enumerate(polygons):
        color = random.choice(MASK_COLORS)

        points = [tuple(p) for p in polygon]
        xmin, ymin, xmax, ymax = _get_bbox_from_points(points)
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

        if with_bbox:
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=1, fill=bbox_color)

        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        if name_list is not None:
            name = name_list[ix]

            if name in color_map_history:
                color = color_map_history[name]
            elif color_map is not None and name in color_map:
                color = color_map[name]
                color = color if len(color) == 4 else color + (round(255 * alpha),)
                color_map_history[name] = color
            else:
                color = color + (round(255 * alpha),)
                color = color if len(color) == 4 else color + (round(255 * alpha),)
                color_map_history[name] = color

            display_str_heights = font.getsize(name)[1]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (1 + 2 * 0.05) * display_str_heights

            if top > total_display_str_height:
                text_bottom = top
            else:
                text_bottom = bottom + total_display_str_height

            text_width, text_height = font.getsize(name)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                  text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                name,
                fill=text_color,
                font=font)
            text_bottom -= text_height - 2 * margin

        draw.polygon(points, fill=color, outline=outline)

    np.copyto(image, np.array(pil_image.convert('RGB')))

    if save_path is not None:
        pil_image.save(save_path)

    if is_show:
        imshow(image)

    return image
