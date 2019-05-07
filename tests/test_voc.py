from icv.data.voc import Voc
import random

if __name__ == '__main__':
    voc = Voc("/Users/rensike/Files/temp/voc_tiny",mode="segment")

    print("voc ids:",voc.ids)

    voc.vis("2007_009322",show=True)