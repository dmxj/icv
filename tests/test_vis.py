# -*- coding: UTF-8 -*-
from icv.image.vis import imdraw_polygons,imshow,imdraw_bbox,imshow_bboxes

if __name__ == '__main__':
    # p = [354.01, 185.07, 366.31, 185.07, 368.99, 175.98, 371.13, 160.47, 377.55, 155.65, 381.83, 147.63, 371.13, 133.19, 369.52, 123.56, 360.43, 133.73, 348.66, 141.21, 326.2, 149.77, 327.27, 151.38, 330.48, 160.47, 330.48, 168.49, 334.76, 175.98, 337.97, 183.47, 354.55, 186.68]
    # polygons = []
    # for i in range(len(p)):
    #     x = p[i]
    #     if i % 2 == 0:
    #         polygons.append((x,p[i+1]))
    #
    # image = imdraw_polygons("/Users/rensike/Files/temp/coco_tiny/val/000000136355.jpg",[polygons])
    #
    # imshow(image)

    imshow_bboxes("/Users/rensike/Files/data/images/boy.jpg",bboxes=[[69,27,133,395]],is_show=True)
