# -*- coding: UTF-8 -*-
from icv.utils import is_valid_url
from urllib import request
from urllib.error import URLError,HTTPError
from urllib.parse import urlencode
from .response import IcvHttpResponse

class IcvHttpRequest(request.Request):
    method = "GET"
    url = ""
    data = {}
    params = {}
    headers = {}
    timeout = 3000

    def __init__(self,method,url,data=None,params=None,headers=None,timeout=3000):
        assert is_valid_url(url)
        self.method = method
        self.url = url
        self.data = data
        self.params = params if params else {}
        self.headers = headers if headers else {}
        self.timeout = timeout

        if self.params is not None:
            params = urlencode(self.params)
            self.url = "?".join([self.url,params])

        super(IcvHttpRequest,self).__init__(self.url,data=self.data,headers=self.headers,method=self.method)

    def __repr__(self):
        return "url:{}\nmethod:{}\ndata:{}\nparams:{}\nheaders:{}\ntimeout:{}".format(
            self.url,self.method,self.data,self.params,self.headers,self.timeout
        )

    def send(self):
        req = request.Request(self.url,data=self.data,headers=self.headers,method=self.method)
        response = IcvHttpResponse(content="",status_code=0)
        try:
            res = request.urlopen(req,timeout=self.timeout)
            response.content = res.read().decode("utf-8")
            response.status_code = 200
            response.headers = res.info()
        except HTTPError as e:
            if hasattr(e,"headers"):
                response.headers = e.headers
            if hasattr(e, 'reason'):
                response.error = e.reason

            response.content = str(e)
            response.status_code = 500
            if hasattr(e, 'code'):
                response.status_code = e.code

        except URLError as e:
            if hasattr(e,"headers"):
                response.headers = e.headers
            if hasattr(e, 'reason'):
                response.error = e.reason

            response.content = str(e)
            response.status_code = 404
            if hasattr(e, 'code'):
                response.status_code = e.code

        return response




