# -*- coding: UTF-8 -*-
from .status_code import HTTP_STATUS_CODES
import cgi

class Context(object):
    protocol = "http"
    ip = "0.0.0.0"
    path = "/"
    method = ""
    headers = {}
    status = ()
    requestvars = cgi.FieldStorage()
    output = ""

    def header(self,field,value):
        if field.lower() == "content-type":
            value += ";charset=utf-8"
        self.headers[field] = value

    def getheaders(self):
        return self.headers

    def set_status(self,stat):
        self.status = str(stat)

    def redirect(self,location):
        """Sets the status to return to the client as 301 Moved Permanently.

            @param location: *str* The location to redirect to.

            @return: None
        """
        self.set_status(HTTP_STATUS_CODES[301][0])
        self.header("Location", location)

    def get(self,varname, default=None):
        """Returns the given HTTP parameter.
            @param varname: *str* The name of the HTTP parameter that should be
            returned.
            @param default: *object* The object that should be returned if the HTTP
            parameter does not exist.
            @return: The value of the HTTP parameter OR if provided, the value of
            default OR if default is not provided, None.
        """
        return self.requestvars.getfirst(varname, default=default)

    def getall(self,**kwargs):
        """Returns the given HTTP parameter.

            @param **kwargs: *kwargs* The "name = default" pairs of the HTTP
            parameters that should be returned.

            @return: The list of values of the HTTP parameter OR the value of
            default.
        """
        # Get the params
        http_params = []
        for key, val in kwargs:
            http_params.append(self.get(key, val))

        # return params
        return http_params

    def clear(self):
        self.protocol = "http"
        self.ip = "0.0.0.0"
        self.path = "/"
        self.method = ""
        self.headers = {}
        self.status = ()
        self.requestvars = cgi.FieldStorage()
        self.output = ""
