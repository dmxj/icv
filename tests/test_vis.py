# -*- coding: UTF-8 -*-
from icv.image.vis import imdraw_polygons, imshow, imdraw_bbox, imshow_bboxes
from icv.vis import draw_line, draw_bar
import matplotlib.pyplot as plt

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

    # imshow_bboxes("/Users/rensike/Files/data/images/boy.jpg",bboxes=[[69,27,133,395]],is_show=True)

    # draw_line(
    #     y_data=[[0.789,0.832,0.889,0.874,0.873,0.888,0.868],[0.092*5,0.147*5,0.188*5,0.168*5,0.218*5,0.180*5,0.186*5]],
    #     x_label="model",
    #     y_label="mAP and time(5s)",
    #     x_ticklabels=["frcnn_r50","+fpn","+fpn+se","+fpn+dcn","+fpn+se+dcn","+fpn+dcn+gn","+fpn+se+ms"],
    #     legends=["mAP","inference time"],
    #     linestyle="-.",
    #     save_path="./benchmark.png",
    #     show_value=True,
    #     title="frcnn benchmark for different feature extractor network",
    #     color=["r","b"]
    # )

    # draw_bar(
    #     data_list=[[0.789, 0.832, 0.889, 0.874, 0.873, 0.888, 0.868],
    #                [0.092, 0.147, 0.188, 0.168, 0.218, 0.180, 0.186]],
    #     legends=["mAP", "inference time"],
    #     save_path="./benchmark_map_time.png",
    #     x_ticklabels=["frcnn_r50", "+fpn", "+fpn+se", "+fpn+dcn", "+fpn+se+dcn", "+fpn+dcn+gn", "+fpn+se+ms"],
    #     title="mAP and infer time for different feature extractor network",
    # )

    # draw_line(
    #     y_data=[[86,83.67,80.33],[93.67,88.67,88],[97.67,95.67,94.67],[93.33,87.67,89],[95,86,86.33]],
    #     x_ticklabels=["epoch 1","epoch 1","","","","epoch 3","","","","epoch 6"],
    #     y_label="non-defective-ratio",
    #     y_line_values=95,
    #     save_path="./non-defective-ratio.png",
    #     title="non-defective-ratio for pure increment train",
    #     legends=["Incr 3","Incr 6","Incr 10","Incr 15","Incr 20"],
    #     color=['black','blue','darkgreen','gold','red']
    # )

    # draw_line(
    #     y_data=[[73.33,72.67,74],[71.67,69.67,70.33],[46,50,50.67],[72,79,78.67],[64,70,69]],
    #     x_ticklabels=["epoch 1","epoch 1","","","","epoch 3","","","","epoch 6"],
    #     y_label="recall",
    #     y_line_values=73,
    #     save_path="./recall.png",
    #     title="recall for pure increment train",
    #     legends=["Incr 3","Incr 6","Incr 10","Incr 15","Incr 20"],
    #     color=['black','blue','darkgreen','gold','red']
    # )

    # draw_line(
    #     y_data=[[100,100,100],[100,100,100],[70,80,80],[86.67,100,100],[85,100,100]],
    #     x_ticklabels=["epoch 1","epoch 1","","","","epoch 3","","","","epoch 6"],
    #     y_label="accuracy",
    #     save_path="./accuracy.png",
    #     title="accuracy for pure increment train",
    #     legends=["Incr 3","Incr 6","Incr 10","Incr 15","Incr 20"],
    #     color=['black','blue','darkgreen','gold','red']
    # )

    # img = imdraw_bbox("./od_infer_result.jpg",10.0,10.0,200.0,200.0,display_str="love")
    # print(img.shape)
    # imshow(img)

    # 横坐标
    xaxis = ["method 1", "method 2", "method 3", "method 4", "method 5"]

    # 纵坐标良率
    liang_ratio_incr3 = [0.98, 0.99, 0.98, 1, 1]
    liang_ratio_incr6 = [0.96,0.98,1,1,0.99]
    liang_ratio_incr10 = [0.99,1,0.99,0.98,1]
    liang_ratio_incr15 = [0.99,0.98,1,1,0.99]
    liang_ratio_incr18 = [0.99,0.98,0.99,0.99,0.97]

    # 纵坐标召回
    recall_incr3 = [0.77, 0.77, 0.82, 0.6, 0.73]
    recall_incr6 = [0.86,0.83,0.73,0.57,0.79,]
    recall_incr10 = [0.76,0.72,0.77,0.65,0.74]
    recall_incr15 = [0.82,0.83,0.78,0.61,0.75]
    recall_incr18 = [0.81,0.85,0.79,0.73,0.82]

    # 纵坐标准率
    accu_incr3 = [1.0,1.0,1.0,1.0,1.0]
    accu_incr6 = [1.0,1.0,1.0,1.0,1.0]
    accu_incr10 = [1.0,1.0,1.0,1.0,1.0]
    accu_incr15 = [1.0,0.93,1.0,1.0,0.93]
    accu_incr18 = [1.0,1.0,1.0,1.0,1.0]

    # titles
    title_incr3 = "target for increment 3 badcase"
    title_incr6 = "target for increment 6 badcase"
    title_incr10 = "target for increment 10 badcase"
    title_incr15 = "target for increment 15 badcase"
    title_incr18 = "target for increment 18 badcase"

    x = list(range(len(xaxis)))
    total_width, n = 0.9, 3
    width = total_width / n

    plt.bar(x, liang_ratio_incr18, width=width, label="good ratio", alpha=0.9)
    for xi, mAP in zip(x, liang_ratio_incr18):
        plt.text(xi, mAP, "%.2f" % mAP, ha="center", va="bottom", fontsize=11)

    for i in range(len(x)):
        x[i] = x[i] + width + 0.0
    plt.bar(x, recall_incr18, width=width, label="recall", tick_label=xaxis, alpha=0.9, color='red')
    for xi, time in zip(x, recall_incr18):
        plt.text(xi, time, "%.2f" % time, ha="center", va="bottom", fontsize=11)

    for i in range(len(x)):
        x[i] = x[i] + width + 0.0
    plt.bar(x, accu_incr18, width=width, label="accu", tick_label=xaxis, alpha=0.9, color='green')
    for xi, time in zip(x, accu_incr18):
        plt.text(xi, time, "%.2f" % time, ha="center", va="bottom", fontsize=11)

    plt.axhline(y=0.9633, xmin=-1, xmax=1, color="blue", linestyle="--", label="base good ratio")
    plt.axhline(y=0.83, xmin=-1, xmax=1, color="red", linestyle="--", label="base recall")

    plt.title(title_incr18)
    plt.legend()
    plt.savefig("./model_iter_incr18.png")
    plt.show()

