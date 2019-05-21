# -*- coding: UTF-8 -*-
from icv.core.http.client.client import IcvHttpClient
from icv.core.http.methods import HttpMethod

def request(method,url,data=None,params=None,headers=None,timeout=3000):
    method = str(method).upper()
    if method == HttpMethod.POST:
        return do_post(url=url,data=data,params=params,headers=headers,timeout=timeout)
    elif method == HttpMethod.GET:
        return do_get(url=url,params=params,headers=headers,timeout=timeout)
    elif method == HttpMethod.PUT:
        return do_put(url=url,data=data,params=params,headers=headers,timeout=timeout)
    elif method == HttpMethod.DELETE:
        return do_post(url=url,data=data,params=params,headers=headers,timeout=timeout)
    elif method == HttpMethod.OPTIONS:
        return do_post(url=url,data=data,params=params,headers=headers,timeout=timeout)
    else:
        raise Exception("Request Method Is Not Support!")

def do_post(url,data=None,json=False,form=False,params=None,headers=None,timeout=3000):
    http_client = IcvHttpClient(url,data=data,params=params,headers=headers,timeout=timeout)
    return http_client.post(json=json,form=form)

def do_get(url,params=None,headers=None,timeout=3000):
    http_client = IcvHttpClient(url, params=params, headers=headers, timeout=timeout)
    return http_client.get()

def do_put(url,data=None,json=False,form=False,params=None,headers=None,timeout=3000):
    http_client = IcvHttpClient(url,data=data,params=params,headers=headers,timeout=timeout)
    return http_client.put(json=json,form=form)

def do_delete(url,data=None,params=None,headers=None,timeout=3000):
    http_client = IcvHttpClient(url, data=data, params=params, headers=headers, timeout=timeout)
    return http_client.delete()

def do_options(url,data=None,params=None,headers=None,timeout=3000):
    http_client = IcvHttpClient(url, data=data, params=params, headers=headers, timeout=timeout)
    return http_client.options()


