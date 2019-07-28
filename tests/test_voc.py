from icv.data.voc import Voc
from icv.vis import DARK_COLORS

if __name__ == '__main__':
    # voc = Voc("/Users/rensike/Files/temp/voc_tiny",mode="segment")
    #
    # print("voc ids:",voc.ids)

    # voc.vis(random.choice(voc.ids),is_show=True,seg_inbox=False)

    # voc1 = Voc("/Users/rensike/Work/huaxing/huaxing_pascal_v0",mode="segment")
    voc1 = Voc("/Users/rensike/Files/temp/voc_tiny", mode="segment")

    voc1_sub = voc1.sub(10)
    print("voc1_sub.ids:", voc1_sub.ids)

    # voc1.crop_bbox_for_classify("/Users/rensike/Work/icv/test_crop_bbox_for_classify", reset=True, split="trainval",
    #                                 pad=20,resize=(200,200))

    voc1_sub.save("/Users/rensike/Work/icv/voc_sub1_test_save",reset_dir=True,split="val")

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
