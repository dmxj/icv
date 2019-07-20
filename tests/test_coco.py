from icv.data.coco import Coco
import matplotlib.pyplot as plt
import pylab
import skimage.io as io

if __name__ == '__main__':
    import os


    # test_coco = Coco(image_dir="/Users/rensike/Work/huaxing/huaxing_coco_v2/images/val",
    #                  anno_file="/Users/rensike/Work/huaxing/huaxing_coco_v2/annotations/val.json")
    #
    # samples = test_coco.get_samples()
    #
    # sample = samples[3]
    #
    # image_id = sample.fields.id
    #
    # anns = test_coco.coco.imgToAnns[image_id]
    #
    # print(sample.fields.anns)
    # print(sample.bbox_list)

    # I = io.imread(sample.get_field("url"))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # test_coco.showAnns(sample.id)

    # test_coco.vis("/Users/rensike/Work/huaxing/huaxing_coco_v2_val_vis")

    # train_coco = Coco(
    #     image_dir="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0/images/train",
    #     anno_file="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0/annotations/train.json"
    # )
    # train_coco.vis(with_bbox=False,save_dir="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0_vis/")

    test_coco = Coco(
        image_dir="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0/images/test",
        anno_file="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0/annotations/test.json"
    )
    test_coco.vis(with_bbox=False,save_dir="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0_vis/")
