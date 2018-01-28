import re
import requests
from bs4 import BeautifulSoup

#generate all pages url
def generate_allurl(user_in_num):
    url = "http://xa.lianjia.com/ershoufang/pg{}/"
    for url_next in range(1, int(user_in_num)):
        yield url.format(url_next)

#get all pages house url
def get_allurl(generate_allurl):
    get_url = requests.get(generate_allurl)
    if get_url.status_code == 200:
        re_set = re.compile('<li.*?class="clear">.*?<a.*?class="img.*?".*?href="(.*?)"')
        re_get = re.findall(re_set, get_url.text)
        return re_get

#get every houser info
def open_url(re_get):
    res = requests.get(re_get)
    if res.status_code == 200:
        info = {}
        re_digit = re.compile(r'\d+.?\d+')
        soup = BeautifulSoup(res.text, 'lxml')
        # print(soup)
        info['标题'] = soup.select('.main')[0].text
        info['房型'] = soup.select('.mainInfo')[0].text
        info['楼层'] = soup.select('.subInfo')[0].text
        info['面积'] = re.findall(re_digit, soup.select('.area')[0].select('.mainInfo')[0].text)[0]
        info['总价'] = re.findall(re_digit, soup.select('.total')[0].text)[0]
        info['每平方售价'] = re.findall(re_digit, soup.select('.unitPriceValue')[0].text)[0]
        info['所在区域'] = soup.select('.areaName')[0].select('.info')[0].text[:5].split()
        return info

def main():
    user_in_num = 36
    for i in generate_allurl(user_in_num):
        re_gets =  get_allurl(i)
        for re_get in re_gets:
            info = open_url(re_get)
            if float(info['总价'])*0.3 < 21.1:
                print(info)

if __name__ == "__main__":
    main()