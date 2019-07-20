from icv.data.labelme import LabelMe
from glob import glob
import os
import random

if __name__ == '__main__':
    # root_dir = "/Users/rensike/Work/icv/annotation/labelme/segment"
    # image_list = glob(root_dir + "/*.jpg")
    # anno_list = [root_dir + "/%s.json" % os.path.basename(_).rsplit(".",1)[0] for _ in image_list]

    root_dir = "/Users/rensike/Work/huaxing/template"
    image_list = glob(root_dir + "/*.png")
    anno_list = [root_dir + "/%s.json" % os.path.basename(_).rsplit(".", 1)[0] for _ in image_list]

    image_anno_path_list = list(zip(image_list,anno_list))

    labelme = LabelMe(image_anno_path_list)

    # labelme.vis(random.choice(labelme.ids),is_show=True)

    labelme.vis(output_path="/Users/rensike/Workspace/huaxing/tpl")