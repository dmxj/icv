# -*- coding: UTF-8 -*-

class IcvHttpResponse(object):
    content=None
    status_code = 200
    error = None
    headers = {}

    def __init__(self, content=None,status_code=200):
        self.content = content
        self.status_code = status_code

    @property
    def ok(self):
        return self.status_code == 200

    def __repr__(self):
        s = ""
        s += "===content===\n" + str(self.content)
        s += "===status code===\n" + str(self.status_code)
        s += "===error===\n" + str(self.error)
        s += "===headers===\n" + str(self.headers)
        return s
