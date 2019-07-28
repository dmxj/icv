# -*- coding: UTF-8 -*-
from icv.utils import IS_PY3

try:
    if IS_PY3:
        from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
    else:
        from SimpleHTTPServer import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
except ImportError:
    raise ModuleNotFoundError

import io
from urllib.parse import urlparse
from ..exceptions import HttpParseException
import os
import re
from icv.utils import json_encode
from ..status_code import HTTP_STATUS_CODES
from http import HTTPStatus
import socket
from .context import Context


def api(ctx):
    data = ctx.request.data_json
    if data and "name" in data:
        print("yes!!!!!")
    ctx.response.send("post api")


def error(ctx):
    ctx.response.send({"error": "you are wrong"}, 500)


def ok_ke(ctx):
    ctx.response.send("post ok ke")


def hello(ctx):
    ctx.response.send("get hello")


def _init_routes():  # 初始化
    global routes
    global method_routes

    routes = {
        "/api": ("POST", api),
        "/error": (["POST", "PUT"], error),
        "/ok/ke": ("POST", ok_ke),
        "/hello": ("GET", hello),
    }
    method_routes = {
        "POST": [
            ("/api", api),
            ("/ok/ke", ok_ke)
        ],
        "GET": [
            ("/hello", hello)
        ]
    }


def add_route(action, methods, function):
    routes[action] = (methods, function)
    if isinstance(methods, str):
        method_routes[methods].append((action, function))
    for m in list(methods):
        if m not in method_routes:
            method_routes[m] = []
        method_routes[m].append((action, function))


class IcvHandler(SimpleHTTPRequestHandler):
    def _prepare_response(self, status_code=200, content_type="text/plain", **kwargs):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        for k in kwargs:
            self.send_header(k, kwargs[k])
        self.end_headers()

    def render(self, html, enc="utf-8"):
        print("#### render html:", html)
        if os.path.isfile(html):
            encoded = open(html, "r").read().encode(enc, 'surrogateescape')
        else:
            # encoded = html.encode(enc, 'surrogateescape')
            return self.error_404()
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def error_404(self):
        self.error(404, "404 Not Found.")

    def error(self, status_code=500, errMsg=None):
        errMsg = errMsg if errMsg and isinstance(errMsg, str) else "Internal Server Error"
        self.response(errMsg, status_code=status_code)

    def response(self, data, status_code=200, **kwargs):
        if isinstance(data, dict):
            res = json_encode(data).encode("utf-8")
        elif isinstance(data, str):
            res = data.encode("utf-8")
        elif isinstance(data, bytes):
            res = data
        else:
            return self.error()

        content_type = None
        if isinstance(data, str):
            content_type = "text/plain;charset=utf-8"
        elif isinstance(data, dict):
            content_type = "application/json;charset=utf-8"
        else:
            if "Content-Type" in kwargs:
                content_type = kwargs["Content-Type"]
            elif "content-type" in kwargs:
                content_type = kwargs["content-type"]

        if content_type is not None:
            self._prepare_response(status_code=status_code, content_type=content_type, **kwargs)
        else:
            self._prepare_response(status_code=status_code, **kwargs)
        self.wfile.write(res)

    def _build_ctx(self):
        try:
            ctx = Context(self, self.command, dict(self.headers._headers))
            ctx.request.raw_path = self.path
            content_length = int(self.headers["Content-Length"]) if "Content-Length" in self.headers else 0
            post_data = self.rfile.read(content_length)
            ctx.request.data = post_data
            return ctx
        except HttpParseException as e:
            self.error(e.code, e.msg)
            self.finish()
            print("===> HttpParseException:", e.msg)
        except Exception as e:
            self.error()
            print("===> Exception:", e)

    def handle_one_request(self):
        try:
            self.raw_requestline = self.rfile.readline(65537)
            if len(self.raw_requestline) > 65536:
                self.requestline = ''
                self.request_version = ''
                self.command = ''
                self.send_error(HTTPStatus.REQUEST_URI_TOO_LONG)
                return
            if not self.raw_requestline:
                self.close_connection = True
                return
            if not self.parse_request():
                # An error code has been sent, just exit
                return
            mname = 'do_' + self.command
            if not hasattr(self, mname):
                self.send_error(
                    HTTPStatus.NOT_IMPLEMENTED,
                    "Unsupported method (%r)" % self.command)
                return

            up = urlparse(self.path)
            target = self._match(up.path.rstrip("/"))
            if target is None:
                if up.path.rstrip("/") == "":
                    self.response({
                        "up": "true",
                        "method": self.command,
                        "path": up.path,
                        "params": up.params,
                        "query": up.query,
                    })
                    return
                self.send_error(
                    HTTPStatus.NOT_FOUND,
                    "Not Found handler (%r)" % self.path
                )
                return

            ms, func, _ = target
            if isinstance(ms, str) and self.command != ms:
                self.send_error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "method not allowed (%r)" % self.command
                )
                return

            if isinstance(ms, (list, tuple)) and self.command not in ms:
                self.send_error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "method not allowed (%r)" % self.command
                )

            method = getattr(self, mname)
            method(func)
            self.wfile.flush()  # actually send the response if not already done.
        except socket.timeout as e:
            # a read or a write timed out.  Discard this connection
            self.log_error("Request timed out: %r", e)
            self.close_connection = True
            return

    def _match(self, rpath):
        print("now math path:", rpath)
        print("routes:", routes)
        for action in routes:
            handler_method, handler_func = routes[action]
            result = re.compile("^" + str(action) + "$").match(str(rpath))
            if result:
                return handler_method, handler_func, [x for x in result.groups()]
        # No match, return None
        print("=====> not match path:", rpath)
        return None

    def _filter(self, method=None):
        target = self._match(self.path)
        if target is None:
            self.error(404, HTTP_STATUS_CODES[404][0])
            self.finish()
            return None

        if method is None:
            return target

        handler_method, _, _ = target
        if isinstance(handler_method, str):
            if handler_method.upper() != str(method).upper():
                self.error(501, HTTP_STATUS_CODES[501][0])
                self.finish()
                return None
        else:
            if str(method).upper() not in handler_method:
                self.error(501, HTTP_STATUS_CODES[501][0])
                self.finish()
                return None

        return target

    def do_GET(self, handler=None):
        if handler is None:
            self.error(400, "handler not exist!")
            return

        handler(self._build_ctx())

    def do_PUT(self, handler=None):
        if handler is None:
            self.error(400, "handler not exist!")
            return

        handler(self._build_ctx())

    def do_DELETE(self, handler=None):
        if handler is None:
            self.error(400, "handler not exist!")
            return

        handler(self._build_ctx())

    def do_POST(self, handler=None):
        if handler is None:
            self.response("handler not exist!")
            return

        handler(self._build_ctx())
