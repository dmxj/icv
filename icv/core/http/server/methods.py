# -*- coding: UTF-8 -*-

class IcvHttpMethod(object):
    POST = "POST"
    GET = "GET"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"
    CONNECT = "CONNECT"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"
    PATCH = "PATCH"

    @property
    def methods(self):
        return list([
            self.POST,
            self.GET,
            self.PUT,
            self.DELETE,
            self.HEAD,
            self.CONNECT,
            self.OPTIONS,
            self.TRACE,
            self.PATCH,
        ])
