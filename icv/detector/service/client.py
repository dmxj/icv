# -*- coding: UTF-8 -*-
from ...image import imshow, imwrite, imshow_bboxes, immerge
from ...utils import do_post, is_file, is_seq, is_str, labelmap_to_categories, json_decode
import os
import base64

class DetectorClient(object):
    def __init__(self, host, port, secret=None, timeout=3000, categories=None):
        host = host.strip("/")
        self._server_addr = "{}:{}".format(host, port) if str(host).startswith("http://") else "http://{}:{}".format(
            host, port)
        self.rest_url = self._server_addr + "/api"
        self.secret = secret
        self.timeout = timeout

        self.categories = categories
        self.id2cat = None
        self.cat2id = None
        if is_seq(categories) and len(categories) > 0:
            self.id2cat = {ix + 1: cat for ix, cat in enumerate(self.categories)}
            self.cat2id = {cat: ix + 1 for ix, cat in enumerate(self.categories)}

    def _encrypt(self, secret):
        if isinstance(secret, str):
            return base64.b64encode(secret.encode("utf-8"))
        return base64.b64encode(secret)

    def set_categories(self, categories):
        if is_seq(categories) and len(categories) > 0:
            self.categories = list(self.categories)
            self.id2cat = {ix + 1: cat for ix, cat in enumerate(self.categories)}
            self.cat2id = {cat: ix + 1 for ix, cat in enumerate(self.categories)}
        else:
            self.categories = None

    def set_labelmap_path(self, labelmap_path):
        assert is_file(labelmap_path)
        categories = labelmap_to_categories(labelmap_path)
        self.categories = [cat["name"] for cat in categories]
        self.id2cat = {ix + 1: cat for ix, cat in enumerate(self.categories)}
        self.cat2id = {cat: ix + 1 for ix, cat in enumerate(self.categories)}

    def inference(self, image_file, timeout=0, is_show=False, save_path=None, vis_merge=False, vis_merge_origin="x"):
        assert is_file(image_file)
        timeout = timeout if timeout > 0 else self.timeout
        headers = {}
        if self.secret is not None:
            headers["secret"] = self._encrypt(self.secret)

        with open(image_file, "rb") as f:
            base64_data = base64.b64encode(f.read())
            image_encoded = str(base64_data)[2:-1]

        post_data = {
            "image": image_encoded,
        }

        print("ヽ( ^∀^)ﾉ\tinference begin")
        response = do_post(self.rest_url, data=post_data, json=True, headers=headers, timeout=timeout)
        print("ヽ( ^∀^)ﾉ\tinference finish, get response")

        if response.ok:
            try:
                result = json_decode(response.content)
                if result["success"]:
                    result_data = result["data"]
                    if is_str(result_data):
                        result_data = json_decode(result_data.replace("'", "\""))
                    bboxes = result_data["bboxes"]
                    classes = result_data["classes"]
                    scores = result_data["scores"]
                    # TODO: imshow_bboxes过滤分值有错误，待修复
                    # scores = None
                    labels = [self.id2cat[cls] for cls in classes] if self.id2cat is not None else None
                    vis_image = imshow_bboxes(image_file, bboxes, classes=self.categories, labels=labels, scores=scores)

                    if vis_merge:
                        vis_image = immerge([image_file, vis_image], origin=vis_merge_origin)

                    if is_show:
                        imshow(vis_image)
                    if save_path is not None:
                        imwrite(vis_image, save_path)
                    return result["data"], vis_image
                else:
                    print("(´-ι_-｀)\t inference failed! code:(%d) error:(%s)" % (result["code"], result["message"]))
                    return None, None
            except Exception as e:
                print("(థฺˇ౪ˇథ)\tparse result error: ", str(e))
                return None, None
        else:
            print("(థฺˇ౪ˇథ)\tinference error: ", response.error)
            return None, None

    def inference_batch(self, image_file_list, timeout=0, save_dir=None, vis_merge=False, vis_merge_origin="x"):
        for image_file in image_file_list:
            assert is_file(image_file), "image is not exist: {}".format(image_file)

        timeout = timeout if timeout > 0 else self.timeout
        headers = {}
        if self.secret is not None:
            headers["secret"] = self._encrypt(self.secret)

        image_encoded_list = []
        for image_file in image_file_list:
            with open(image_file, "rb") as f:
                base64_data = base64.b64encode(f.read())
                image_encoded = str(base64_data)[2:-1]
                image_encoded_list.append(image_encoded)

        post_data = {
            "image": image_encoded_list,
        }

        response = do_post(self.rest_url, data=post_data, json=True, headers=headers, timeout=timeout)

        if response.ok:
            try:
                result = json_decode(response.content)
                if result["success"]:
                    result_data = result["data"]
                    vis_image_list = []
                    for image_file, data in zip((image_file_list, result_data)):
                        bboxes = data["bboxes"]
                        classes = data["classes"]
                        # scores = result["data"]["scores"]
                        # TODO: imshow_bboxes过滤分值有错误，待修复
                        scores = None
                        labels = [self.id2cat[cls] for cls in classes] if self.id2cat else None
                        vis_image = imshow_bboxes(image_file, bboxes, classes=self.categories, labels=labels,
                                                  scores=scores)

                        if vis_merge:
                            vis_image = immerge([image_file, vis_image], origin=vis_merge_origin)

                        if save_dir is not None:
                            imwrite(vis_image, os.path.join(save_dir, os.path.basename(image_file)))
                    return result_data, vis_image_list
                else:
                    print("(´-ι_-｀)\t inference failed! code:(%d) error:(%s)" % (result["code"], result["message"]))
                    return None, None
            except Exception as e:
                print("(థฺˇ౪ˇథ)\tparse result error: ", str(e))
                return None, None
        else:
            print("(థฺˇ౪ˇథ)\tinference error: ", response.error)
            return None, None
