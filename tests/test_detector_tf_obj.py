# -*- coding: UTF-8 -*-
from icv.detector import TfObjectDetector

if __name__ == '__main__':
    detector = TfObjectDetector(
        model_path="/Users/rensike/Resources/models/tensorflow/frozen_inference_graph.pb",
        labelmap_path="/Users/rensike/Workspace/themis_inference_service/config/mscoco_label_map.pbtxt"
    )

    detector.start_server(9527,secret="leilei")
    # detector.inference("/Users/rensike/Files/data/images/street.jpg",is_show=True)

