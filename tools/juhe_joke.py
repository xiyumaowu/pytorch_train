import urllib,json
from urllib.parse import urlencode
import urllib.request

AppKey = "b8eff1f6cc0c597d65cd7fd007b09b65"

def request2(appkey, m="GET"):
    url = "http://v.juhe.cn/joke/content/text.php"
    params = {
        "page": 1,
        "pagesize": 20,
        "key": AppKey
    }
    params = urlencode(params)
    if m == "GET":
        f = urllib.request.urlopen("%s?%s" %(url,params))
    else:
        f = urllib.request.urlopen(url, params)

    content = f.read()
    res = json.loads(content)
    jokes = ''
    if res:
        err_code = res["error_code"]
        if err_code == 0:
            for datas in res["result"]["data"]:
                jokes = jokes + datas["content"] + '\n'

        else:
            print("%s:%s" %(res["error_code"], res["reason"]) )
    else:
        print("request api error!")
    return jokes


print(request2(AppKey))