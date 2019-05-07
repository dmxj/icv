# -*- coding: UTF-8 -*-
from icv.utils import IS_PY3
try:
    if IS_PY3:
        from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
    else:
        from SimpleHTTPServer import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
except:
    raise ModuleNotFoundError

from wsgiref.handlers import CGIHandler
import http
import re
import json
from .status_code import HTTP_STATUS_CODES
from .methods import IcvHttpMethod

class IcvHandler(BaseHTTPRequestHandler):
    def __init__(self,routes=None):
        super(IcvHandler,self).__init__(self.request,self.client_address,self.server)
        self.routes = routes

        self.method_routes = {method: [] for method in list(IcvHttpMethod.methods)}
        for action in self.routes:
            (method,func) = self.routes[action]
            self.method_routes[method].append((action,func))

    def _set_headers(self,status_code=200,content_type="application/json",**kwargs):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        for k in kwargs:
            self.send_header(k,kwargs[k])
        self.end_headers()

    def error(self,status_code=500,errMsg=None):
        errMsg = errMsg if errMsg and isinstance(errMsg,str) else "Internal Server Error"
        self.response(errMsg,status_code=status_code)

    def response(self,data,status_code=200,content_type="application/json",**kwargs):
        if isinstance(data,dict):
            res = json.dumps(data).encode("utf-8")
        elif isinstance(data,str):
            res = json.dumps(data).encode("utf-8")
        elif isinstance(data,bytes):
            res = data
        else:
            return self.error()

        self._set_headers(status_code=status_code,content_type=content_type,**kwargs)
        self.wfile.write(res)

    def _match(self, rpath):
        for action in self.routes:
            handler_method,handler_func = self.routes[action]
            result = re.compile("^" + str(action) + "$").match(str(rpath))
            if result:
                return handler_method,handler_func,[x for x in result.groups()]
        # No match, return None.
        return None

    def _filter(self,method=None):
        target = self._match(self.path)
        if target is None:
            self.error(404,HTTP_STATUS_CODES[404][0])
            return None

        if method is None:
            return target

        handler_method,_,_ = target
        if isinstance(handler_method,str):
            if handler_method.upper() != str(method).upper():
                self.error(501, HTTP_STATUS_CODES[501][0])
                return None
        else:
            if str(method).upper() not in handler_method:
                self.error(501, HTTP_STATUS_CODES[501][0])
                return None

        return target

    def do_GET(self):
        target = self._filter()
        if target:
            pass


    def do_PUT(self):
        self.response({"hello":"put method hello world"})
        print("end do put")

    def do_DELETE(self):
        self.response({"name":"rsk hahaha"})

    def do_POST(self):
        # print("do post request path:",self.request.path)
        print("do post request:",self.request)
        # print("do post request data_get:",self.request.data_get)
        # print("do post request data_json:",self.request.data_json)
        # print("do post request data_post:",self.request.data_post)
        print("do post path:",self.path)
        print("do post headers:",self.headers["author"])
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        print("post data from client:",post_data)

        try:
            post_data = str(post_data.decode("utf-8")).replace("\n","").replace("\t","").replace("\r","")
            print("post_data:",post_data)
            req_body = json.loads(str(post_data))
            for req_key in req_body:
                print("{}==={}".format(req_key,req_body[req_key]))
        except Exception as e:
            print("get req body error:",e)

        response = {
            'status':"SUCCESS",
            "data":"server got your post data"
        }

        self._set_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))



