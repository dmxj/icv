# -*- coding: UTF-8 -*-
try:
    # Python 2.x
    from SocketServer import ThreadingMixIn
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from BaseHTTPServer import HTTPServer
except ImportError:
    # Python 3.x
    from socketserver import ThreadingMixIn
    from http.server import SimpleHTTPRequestHandler, HTTPServer

class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass

class IcvServer(object):
    def __init__(self,host="0.0.0.0",port=9527):
        self.host = host
        self.port = port
        self.server = None
        self.handler = IcvServer

    def run(self,handler_class):
        self.server = ThreadingSimpleServer((self.host, self.port), handler_class)
        self.server.serve_forever()

