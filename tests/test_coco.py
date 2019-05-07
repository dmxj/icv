from icv.data.coco import Coco
import matplotlib.pyplot as plt
import pylab
import skimage.io as io

if __name__ == '__main__':
    test_coco = Coco(image_dir="/Users/rensike/Files/temp/coco_tiny/test/",
                     anno_file="/Users/rensike/Files/temp/coco_tiny/annotations/instances_test.json")

    samples = test_coco.get_samples()

    sample = samples[0]

    image_id = sample.fields.id

    anns = test_coco.coco.imgToAnns[image_id]

    print(sample.fields.anns)
    print(sample.bbox_list)

    # I = io.imread(sample.get_field("url"))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    test_coco.showAnns(sample.id)
