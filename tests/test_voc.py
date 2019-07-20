from icv.data.voc import Voc
import random

if __name__ == '__main__':
    voc = Voc("/Users/rensike/Files/temp/voc_tiny",mode="segment")

    print("voc ids:",voc.ids)

    # voc.vis(random.choice(voc.ids),is_show=True,seg_inbox=False)

    # voc1 = Voc("/Users/rensike/Work/huaxing/huaxing_pascal_v0",mode="segment")
    voc1 = Voc("/Users/rensike/Files/temp/voc_tiny",mode="segment")

    voc1_sub = voc1.sub(10)
    print("voc1_sub.ids:",voc1_sub.ids)

    voc_concat = voc.concat(voc1_sub,output_dir="/Users/rensike/Work/icv/test_voc_concat")

    print("concat voc finish!")

    voc_concat.vis(output_path="/Users/rensike/Work/icv/test_voc_concat_vis")
