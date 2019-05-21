# -*- coding: UTF-8 -*-
from icv.detector import DetectorClient
from icv.image import imshow

if __name__ == '__main__':
    client = DetectorClient(host="127.0.0.1",port="9528",secret="keke")
    client.set_labelmap_path("/Users/rensike/Workspace/themis_inference_service/config/mscoco_label_map.pbtxt")
    result,image = client.inference(
        "/Users/rensike/Files/data/images/street.jpg",
        is_show=True,
        save_path="/Users/rensike/Files/data/images/street_icv_det.jpg",
        vis_merge=True
    )

    imshow(image)

    # client.inference(
    #     "/Users/rensike/Files/data/images/street2.jpg",
    #     is_show=True,
    #     save_path="/Users/rensike/Files/data/images/street2_icv_det.jpg",
    #     vis_merge=True
    # )
