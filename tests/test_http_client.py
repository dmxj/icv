import requests
from icv.core.http.client.request import IcvHttpRequest
from icv.core.http.client.client import IcvHttpClient

if __name__ == '__main__':
    # requests.Request()
    # r = requests.get("http://httpbin.org/get",params={"author":"rsk"})
    # print(r)

    # req = IcvHttpRequest("GET","http://httpbin.org/get")
    # res = req.send()
    # print(res)
    #
    # print(res.content)

    url = "http://httpbin.org/put"
    client = IcvHttpClient(url,{"lover":"menglei"},params={"age":26})
    response = client.put(json=True)
    print(response)
    print(response.ok)
    if response.ok:
        print("response ok")
        print(response.content)
        # print(response["content"])
    else:
        print("response bad")
        print(response.error)

