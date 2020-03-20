
from icv.core.http.server.simple_http_server import IcvServer
from icv.core.http.server.app import App

if __name__ == '__main__':
    # app = IcvServer()
    # app.run()

    # import requests
    #
    # req = requests.Request("put", "http://httpbin.org/put")
    # prepared = req.prepare()
    # print(prepared.__dict__)

    app = App(port=9898)

    @app.route("/api/v1/predict",["POST","PUT","GET"])
    def love_leilei(ctx):
        print("=====> you are using method:",ctx.request.method)
        ctx.response.send("i love leilei")

    app.start()

