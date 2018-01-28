# _*_ coding: utf-8 _*_
# cities.py
# author: Junliang Zhong

import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://www.lianjia.com"

#get html
html = urlopen(url).read()

#get BeautifulSoup Object, exchange to
bsobj = BeautifulSoup(html, 'html5lib')
#get class='fc-main clear'
city_tags = []
for i in bsobj.find_all('ul', {"class": "clear"}):
    # print(i.findChildren('a'))
    for a in i.findChildren('a'):
        city_tags.append(a)
"""
    city_tags 内数据的格式如下

    <a title="天津房产网" href="https://tj.lianjia.com/">天津</a>
    <a title="青岛房产网" href="https://qd.lianjia.com/">青岛</a>
    ...
"""
# 将第一条数据抽离， 保存在cities.csv 文件中
with open("./cities.csv", 'w') as f:
    write = csv.writer(f)
    for city_tag in city_tags:
        city_name = city_tag.get_text().encode('utf-8').decode('utf-8')
        if city_name is not "":
            city_url = city_tag.get("href").encode('utf-8').decode('utf-8')
            write.writerow((city_name, city_url))
        # print(city_name, city_url)
    f.close()