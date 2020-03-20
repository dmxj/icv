from icv.data.coco import Coco
import matplotlib.pyplot as plt
import pylab
import skimage.io as io
import random

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

    # test_coco = Coco(
    #     image_dir="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0/images/test",
    #     anno_file="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0/annotations/test.json"
    # )
    # test_coco.vis(test_coco.ids[3],is_show=True,save_dir="/Users/rensike/Work/huaxing/poc/poc_wrong_coco_v0_vis/")

    test_coco = Coco(
        image_dir="/Users/rensike/Files/temp/coco_tiny/test",
        anno_file="/Users/rensike/Files/temp/coco_tiny/annotations/instances_test.json"
    )

    # test_coco.vis(random.choice(test_coco.ids),is_show=True,save_dir="/Users/rensike/Work/icv/test_coco_vis")

    test_coco.statistic(print_log=True)

    # test_coco.crop_bbox_for_classify("/Users/rensike/Work/icv/test_crop_bbox_for_classify_coco", reset=True, split="trainval",
    #                                 pad=20)

    test_coco.save("/Users/rensike/Work/icv/test_coco_save_test", reset_dir=True, split="train")

    from icv.data.coco import Coco
    cat_ids = [1,2,3]

    coco_train = Coco(
        image_dir="/path/to/coco/images/train",
        anno_file="/path/to/coco/annotations/train.json"
    )
    coco_train.ids = coco_train.coco.getImgIds(catIds=cat_ids)
    coco_train.save("/path/to/new_coco/",reset_dir=True,split="train")

    coco_val = Coco(
        image_dir="/path/to/coco/images/val",
        anno_file="/path/to/coco/annotations/val.json"
    )
    coco_val.ids = coco_val.coco.getImgIds(catIds=cat_ids)
    coco_val.save("/path/to/new_coco/",split="val")
