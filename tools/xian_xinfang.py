import re
import requests
from bs4 import BeautifulSoup
import json

#generate all pages url
def generate_allurl(user_in_num):
    url = "http://xa.fang.lianjia.com/loupan/pg{}/"
    for url_next in range(1, int(user_in_num)):
        print(url.format(url_next))
        yield url.format(url_next)

#get all pages house url
def get_allurl(generate_allurl):
    get_url = requests.get(generate_allurl)
    # print(get_url)
    if get_url.status_code == 200:
        # re_set = re.compile('<li.*?class="clear">.*?<a.*?class="img.*?".*?href="(.*?)"')
        re_set = re.compile('<a target="_blank" data-xftrack="10138" href="(.*?)" .*?data-el="xinfang">')
        re_get =  re.findall(re_set, get_url.text)
        return re_get

#get every houser info
def open_url(re_get):
    res = requests.get('https://xa.fang.lianjia.com' + re_get)
    info = {}
    if res.status_code == 200:
        re_digit = re.compile(r'\d+.?\d+')
        soup = BeautifulSoup(res.text, 'lxml')
        # print(soup)
        info['标题'] = soup.select('.name-box')[0].text[6:].replace('\n', '')
        # info['房型'] = soup.select('.mainInfo')[0].text
        # info['楼层'] = soup.select('.subInfo')[0].text
        # info['面积'] = re.findall(re_digit, soup.select('.area')[0].select('.mainInfo')[0].text)[0]
        # info['总价'] = re.findall(re_digit, soup.select('.total')[0].text)[0]
        if re.findall(re_digit, soup.select('.jiage')[0].text):
            info['均价'] = re.findall(re_digit, soup.select('.jiage')[0].text)[0]
        else:
            info['均价'] = '待定'
        info['开盘时间']  = soup.select('.when')[0].text[7:].replace('\n', '')
        info['所在区域'] = soup.select('.where')[0].text[6:].replace('\n', '')
        info['链接'] = 'https://xa.fang.lianjia.com' + re_get
    else:
        pass
    return info

def houseinfo_to_json(user_in_num):
    # user_in_num = 2
    json_file = open('xianfang.json','w')
    house_info = []
    for i in generate_allurl(user_in_num):
        re_gets = get_allurl(i)
        for re_get in re_gets:
            info = open_url(re_get)
            # if float(info['每平方售价']) < 10000.0:
            house_info.append(info)
    json.dump({"house_info" :house_info}, json_file, ensure_ascii=False)
    json_file.close()

def main():
    house_info = houseinfo_to_json(97)
    # print(house_info)

if __name__ == "__main__":
    main()