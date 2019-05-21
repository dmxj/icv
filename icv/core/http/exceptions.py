# -*- coding: UTF-8 -*-

class HttpParseException(Exception):
    # Exception raised by http request parser
    def __init__(self,msg,code=500):
        super(HttpParseException, self).__init__(msg)
        self.msg = msg
        self.code = code
