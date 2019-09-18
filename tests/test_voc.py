from icv.data.voc import Voc
from icv.vis import DARK_COLORS

import time
from functools import wraps

def loop_run(max_times=7,max_duration=7,sleep_time=0.2,loop_msg="out of memory",default_return=Exception("device is busy")):
    def run(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            t = 0
            t0 = time.time()
            while True:
                t += 1
                x = None
                try:
                    x = function(*args, **kwargs)
                except RuntimeError as e:
                    if loop_msg not in str(e):
                        return x
                    if t > max_times:
                        return default_return

                    if time.time() - t0 > max_duration:
                        return default_return

                    time.sleep(sleep_time)
                    continue

        return wrapper

    return run


if __name__ == '__main__':
    import random
    voc = Voc("/Users/rensike/Files/temp/voc_tiny",mode="segment")
    #
    # print("voc ids:",voc.ids)

    voc.vis(random.choice(voc.ids),is_show=True)

    # voc1 = Voc("/Users/rensike/Work/huaxing/huaxing_pascal_v0",mode="segment")
    # voc1 = Voc("/Users/rensike/Work/icv/voc_sub1_test_save", mode="segment",split="val")
    # voc1.vis(save_dir="/Users/rensike/Work/icv/voc1_vis", reset_dir=True)

    # voc1_sub = voc1.sub(10)
    # print("voc2_sub.ids:", voc1_sub.ids)
    #
    # voc1_sub.save("/Users/rensike/Work/icv/voc_sub1_test_save",reset_dir=True,split="val")
    #
    # voc1_sub.vis(save_dir="/Users/rensike/Work/icv/voc2_sub_vis", reset_dir=True)

    # voc1.crop_bbox_for_classify("/Users/rensike/Work/icv/test_crop_bbox_for_classify", reset=True, split="trainval",
    #                                 pad=20,resize=(200,200))

    # voc2_sub.save("/Users/rensike/Work/icv/voc_sub1_test_save",reset_dir=True,split="val")

    #
    # voc_concat = voc.concat(voc1_sub,output_dir="/Users/rensike/Work/icv/test_voc_concat",reset=True)
    #
    # print("concat voc finish!")
    #
    # voc1.statistic(print_log=True)

    # voc1_sub.vis(save_dir="/Users/rensike/Work/icv/test_voc_sub_concat_vis")

    # voc_concat.vis(save_dir="/Users/rensike/Work/icv/test_voc_concat_vis")

    # from icv.data.voc import Voc
    # from icv.vis import DARK_COLORS
    # voc_radar = Voc("/Users/rensike/Work/radar/VOC2007")
    # voc_radar.set_colormap(DARK_COLORS)
    # voc_radar.vis(save_dir="/Users/rensike/Work/radar/radar_vis_2",reset_dir=True)

    # import torch
    # try:
    #     torch.cuda.memory_allocated()
    # except RuntimeError as e:
    #     str(e)
    #

