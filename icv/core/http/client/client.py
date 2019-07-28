# -*- coding: UTF-8 -*-
from icv.utils import is_valid_url, json_encode
from ..methods import HttpMethod
from .request import IcvHttpRequest


class IcvHttpClient(object):
    def __init__(self, url, data=None, params=None, headers=None, timeout=3000):
        if not is_valid_url(url):
            raise Exception("request url is invalid")

        self.method = "GET"
        self.url = url
        self.data = data
        self.params = params if params else {}
        self.headers = headers if headers else {}
        self.timeout = timeout

    def set_method(self, method):
        method = str(method).upper()
        if method not in HttpMethod:
            raise Exception("request method invalid")
        self.method = method

    def set_headers(self, headers):
        self.headers = headers if headers else {}

    def header(self, key, value):
        self.headers[key] = value

    def set_data(self, data):
        self.data = data

    def set_params(self, params):
        self.params = params if params else {}

    def set_timeout(self, timeout):
        self.timeout = timeout

    def do_request(self):
        req = IcvHttpRequest(self.method, self.url, data=self.data, params=self.params, headers=self.headers,
                             timeout=self.timeout)
        return req.send()

    def post(self, json=False, form=False):
        if json:
            if isinstance(self.data, dict):
                self.data = json_encode(self.data).encode("utf-8")
            self.header("Content-Type", "application/json")
        elif form:
            if isinstance(self.data, dict):
                self.data = "&".join(["{}={}".format(k, self.data[k]) for k in self.data])
            self.header("Content-Type", "application/x-www-form-urlencoded")

        self.method = HttpMethod.POST
        return self.do_request()

    def get(self):
        self.method = HttpMethod.GET

        return self.do_request()

    def put(self, json=False, form=False):
        if json:
            if isinstance(self.data, dict):
                self.data = json_encode(self.data).encode("utf-8")
            self.header("Content-Type", "application/json")
        elif form:
            if isinstance(self.data, dict):
                self.data = "&".join(["{}={}".format(k, self.data[k]) for k in self.data])
            self.header("Content-Type", "application/x-www-form-urlencoded")

        self.method = HttpMethod.PUT
        return self.do_request()

    def delete(self):
        self.method = HttpMethod.DELETE
        return self.do_request()

    def options(self):
        self.method = HttpMethod.OPTIONS
        return self.do_request()

    def __repr__(self):
        return "<IcvHttpClient> [{}] {}".format(self.method, self.url)
