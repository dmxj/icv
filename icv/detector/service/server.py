# -*- coding: UTF-8 -*-
import numpy as np
import os
import base64
from ...utils import json_encode
from ...core.http.methods import HttpMethod
from ..process.detect_processor import DetectionProcessor
from ..result import DetectionResult
from .. import detector
from bottle import template, Bottle, request, static_file, tob
from io import BytesIO
import tempfile
import bottle

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 * 1024

ERROR_AUTH = 100
ERROR_JSON_PARSER = 101
ERROR_PARAM_LACK = 102
ERROR_IMAGE_PARSER = 103
ERROR_IMAGE_FORMAT = 104
ERROR_DETECT_ERROR = 105

RESOURCE_ROOT = os.path.dirname(detector.__file__) + "/resource"


class DetectorServer(object):
    html_path = RESOURCE_ROOT + "/serve.html"
    static_root = RESOURCE_ROOT + "/static"
    server_name = "Detector Server"
    default_name = "Icv Predictor"

    def __init__(
            self,
            detector,
            secrect=None,
            open_web=True,
            name="",
            default_params_score_thr=0.5):
        self.detector = detector
        self.secrect = secrect
        self.open_web = open_web
        self.app = Bottle()
        self.name = self.default_name if name == "" else ""
        self.default_params_score_thr = default_params_score_thr
        self._fake_req_env()

    def _fake_req_env(self):
        body = "abc"
        request.environ['CONTENT_LENGTH'] = str(len(tob(body)))
        request.environ['wsgi.input'] = BytesIO()
        request.environ['wsgi.input'].write(tob(body))
        request.environ['wsgi.input'].seek(0)

    def _json_response(self, code, message="", success=False, data=None, **kwargs):
        if data is not None:
            res = {
                "code": code,
                "message": message,
                "success": success,
                "data": json_encode(data)
            }
        else:
            res = {
                "code": code,
                "message": message,
                "success": success,
            }

        for k in kwargs:
            res[k] = kwargs[k]

        return res

    def _decrypt(self, encoded):
        return str(base64.b64decode(encoded).decode("utf-8"))

    def _inference_server(self):
        request_data = request.json
        request_headers = request.headers
        try:
            if self.secrect is not None:
                if "secret" in request_headers and self._decrypt(request_headers["secret"]) == self.secrect:
                    pass
                elif "Secret" in request_headers and self._decrypt(request_headers["Secret"]) == self.secrect:
                    pass
                elif "HTTP_SECRET" in request_headers and self._decrypt(request_headers["HTTP_SECRET"]) == self.secrect:
                    pass
                else:
                    return self._json_response(ERROR_AUTH, "Authentication failed.")
        except:
            return self._json_response(ERROR_AUTH, "Authentication failed.")

        if request_data is None:
            return self._json_response(ERROR_JSON_PARSER, 'bad request, json parse failed.')

        if "image" not in request_data:
            return self._json_response(ERROR_PARAM_LACK, "image param can't empty.")

        request_image = request_data["image"]
        try:
            if isinstance(request_image, list):
                image_np = [_ for _ in DetectionProcessor.pre_process(request_image)]
            else:
                image_np = DetectionProcessor.pre_process(request_image)
                if not isinstance(image_np, np.ndarray):
                    return self._json_response(ERROR_IMAGE_FORMAT,
                                               "parse input image failed, please check input format.")
        except Exception as e:
            return self._json_response(ERROR_IMAGE_PARSER, 'bad request, input image parse failed:' + str(e))

        try:
            if isinstance(image_np, list):
                outputs = self.detector.inference_batch(image_np)
            else:
                outputs = self.detector.inference(image_np)
        except Exception as e:
            return self._json_response(ERROR_DETECT_ERROR,
                                       'predict failure, {0}:{1}'.format(e.__class__.__name__, str(e)))

        outputs = DetectionProcessor.post_process(outputs)
        return self._json_response(0, "success", True, outputs)

    def _inference_page(self):
        return template(
            self.html_path,
            name=self.name,
            default_params_score_thr=self.default_params_score_thr
        )

    def _inference_upload_for_predict(self):
        score_thr = request.forms.get("score_thr")

        if score_thr != "":
            score_thr = float(score_thr)
        else:
            score_thr = -1

        upload = request.files.get('uploadfile')
        name, ext = os.path.splitext(upload.filename)
        ext = ext.lower()
        if ext not in ('.png', '.jpg', '.jpeg', ".tif", ".bmp"):
            return self._json_response(ERROR_IMAGE_FORMAT, "Image File Is Invalid!")

        upload.save(self.upload_dir, overwrite=True)
        det_img_name = "{}_det{}".format(name, ext)
        det_img_path = os.path.join(self.upload_dir, det_img_name)
        upload_img_path = os.path.join(self.upload_dir, upload.filename)
        inference_result = self.detector.inference(upload_img_path, save_path=det_img_path, score_thr=score_thr)
        if isinstance(inference_result, DetectionResult):
            return self._json_response(
                code=ERROR_DETECT_ERROR,
                success=True,
                data={
                    "最大置信度": "%.4f" % inference_result.topk(1)[0][1],
                    "检测结果数": inference_result.length,
                    "预测时间": "%.4fs" % inference_result.det_time,
                },
                image_origin="/img/%s" % upload.filename,
                image_detect="/img/%s" % det_img_name,
            )

        return self._json_response(ERROR_DETECT_ERROR, "Predict Error!")

    def start_server(self, port, upload_dir=None, debug=True):
        self.detector._warmup()

        if upload_dir is None:
            upload_dir = tempfile.mkdtemp()

        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        if not os.path.exists(upload_dir):
            raise Exception("Can not create upload directory!")

        self.upload_dir = upload_dir

        self.app.route("/static/:path#.+#", HttpMethod.GET, lambda path: static_file(path, self.static_root))
        self.app.route("/img/:filename", HttpMethod.GET, lambda filename: static_file(filename, upload_dir))
        self.app.route("/upload", HttpMethod.POST, self._inference_upload_for_predict)
        self.app.route("/api", HttpMethod.POST, self._inference_server)
        if self.open_web:
            self.app.route("/page", HttpMethod.GET)(self._inference_page)
            self.app.route("/", HttpMethod.GET)(self._inference_page)

        self.app.run(host="0.0.0.0", port=port, debug=debug)
