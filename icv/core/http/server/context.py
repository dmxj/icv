# -*- coding: UTF-8 -*-
from ..status_code import HTTP_STATUS_CODES
import cgi
from urllib.parse import urlparse,unquote_plus,parse_qs
from icv.utils import EasyDict
from ..methods import HttpMethod
from ..exceptions import HttpParseException
import json

class HttpRequest(object):
    method = ""
    headers = EasyDict()
    _raw_path = ""
    path = ""
    params = EasyDict()
    query = EasyDict()
    _data = None
    data_post = None
    data_json = None
    requestvars = cgi.FieldStorage()

    def __init__(self,method,headers):
        assert method in HttpMethod.methods
        self.method = method
        self.headers = EasyDict(headers) if isinstance(headers,dict) else EasyDict()

    @property
    def content_type(self):
        if "Content-Type" in self.headers:
            return str(self.headers["Content-Type"]).lower()

        if "content-type" in self.headers:
            return str(self.headers).lower()

        return ""

    @property
    def raw_path(self):
        return self._raw_path

    @raw_path.setter
    def raw_path(self,raw_path):
        self._raw_path = raw_path
        self._parse_url()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,data):
        self._data = data.decode("utf-8") if isinstance(data,bytes) else data
        if isinstance(self._data, str):
            try:
                self.data_post = json.loads(self._data)
                self.data_json = self.data_post
            except:
                if self._data != "":
                    d = self._build_params(self._data)
                    self.data_post = d if len(d) > 0 else self._data
                else:
                    self.data_post = self._data
                    self.data_json = {}
        else:
            if isinstance(self._data,dict):
                self.data_json = self._data
            self.data_post = self._data

        if "application/json" in self.content_type:
            if self.data_json is None:
                raise HttpParseException("request format error! it's not a valid json format!",400)

    def _build_params(self,params_str):
        if not isinstance(params_str,str):
            return EasyDict()
        try:
            return EasyDict(dict(unquote_plus(y).split('=') for y in params_str.split('&')))
        except:
            return EasyDict()

    def _parse_url(self):
        up = urlparse(self.raw_path)
        self.path = up.path
        self.query = self._build_params(up.query)
        self.params = self._build_params(up.params)

    def __str__(self):
        s = ""
        for attr in self.__dict__:
            s += "{}={}\n".format(attr,self.__dict__[attr])
        return s


class HttpResponse(object):
    def __init__(self,handler):
        self._handler = handler
        self.headers = {}
        self.status = ()

    def add_header(self,header_key,header_value):
        self.headers[header_key] = header_value

    def set_status(self,stat):
        self.status = str(stat)

    def send(self,data,code=200,headers=None):
        headers = headers.update(self.headers) if headers else {}
        return self._handler.response(data, code, **headers)

    def render(self, html, enc="utf-8"):
        return self._handler.render(html, enc)

    def redirect(self,location):
        self.set_status(HTTP_STATUS_CODES[301][0])
        self.add_header("Location", location)

    def __str__(self):
        s = ""
        for attr in self.__dict__:
            s += "{}={}\n".format(attr,self.__dict__[attr])
        return s

class Context(object):
    def __init__(self,handler,req_method,req_headers):
        self.request = HttpRequest(req_method,req_headers)
        self.response = HttpResponse(handler)

    def __repr__(self):
        return "--<Context>--\nrequest=[%s]\nresponse=[%s]" % (str(self.request),str(self.response))

    def __str__(self):
        return self.__repr__()