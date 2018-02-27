# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
import re
import  requests
import sys
from email.mime.text import MIMEText
import smtplib
from email.header import Header

class LianjiaSpider(scrapy.Spider):
    name = 'lianjia'
    allowed_domains = ['xa.lianjia.com']
    start_url = 'http://xa.lianjia.com/ershoufang/'
    start_html = requests.get(start_url).text
    start_urls = []
    re_set = re.compile('<div class="page-box house-lst-page-box".*?"totalPage":(.*?),"curPage".*?>') #find ershoufang info totalPages number
    max_page_num = int(re.findall(re_set, start_html)[0]) #ger ershoufang pages number
    bashurl = 'http://xa.lianjia.com/ershoufang/pg{}'
    for i in range(1, max_page_num+1):
        start_urls.append(bashurl.format(i)) # get start urls
    # start_urls = ['http://xa.lianjia.com/ershoufang/pg1']
    def __init__(self):
        self.count = 0
        self.round_num = 0
        self.html_code = ''
        try:
            with open('ershoufang.csv', 'a+') as self.csv:
                csv_tilte = '编号,' + '标题,' + '详细信息,' + '位置信息,' + '关注,' + '总价,' + '单价,' + '链接\n'
                self.csv.write(csv_tilte)
        except Exception as e:
            print(e)
            sys.exit()


    def parse(self, response):
        self.round_num += 1
        html = BeautifulSoup(response.text, 'lxml').select('.sellListContent')[0].select('li .clear')
        re_digit = re.compile('\d+.?\d+')
        re_area = re.compile('厅.*?(.*?)平米')

        try:
            csv = open('ershoufang.csv', 'a+')
        except Exception as e:
            print(e)
            sys.exit()
        for html_i in html:
            title = html_i.select('.title')[0].text
            houselink = html_i.select('.title')[0].find('a').get('href')
            houseinfo = html_i.select('div .houseInfo')[0].text
            f_housearea = re.findall(re_digit, re.findall(re_area, houseinfo)[0])[0]
            positioninfo = html_i.select('div .positionInfo')[0].text
            followinfo = html_i.select('.followInfo')[0].text
            f_totalprice = re.findall(re_digit,html_i.select('div .totalPrice')[0].text)[0]
            f_unitprice = re.findall(re_digit, html_i.select('div .unitPrice')[0].text)[0]
            if float(f_totalprice)< 80.0 and float(f_totalprice) > 60.0:
                if float(f_unitprice) > 8000.0 and float(f_unitprice) < 12000.0:
                    self.count += 1
                    data =str(self.count) + ',' +title.replace(',', '') + ',' + houseinfo + ',' + positioninfo + ',' + followinfo + ',' + str(f_totalprice) + ',' + str(f_unitprice) + ',' + houselink + '\n'
                    # csv.write(data)
                    self.html_code = self.html_code + str(html) + '\n'
                    print(data)
                    # print(data)
        csv.close()
        # print(self.round_num, self.max_page_num)
        # if self.html_code and self.round_num == self.max_page_num:
            # print(html_code)
            # self.mail_to_html(str(self.html_code))

    def mail_to_html(self, html_code):
        mailInfo = {
            "from": '357594634@qq.com',
            "to": 'junliang_zhong@ericsson.com',
            "hostname": 'smtp.qq.com',
            "username": '357594634@qq.com',
            "password": 'sycvoaenwndabibf',
            "mailsubject": '链家二手房',
            "mailtext": html_code,
            "mailencoding": 'utf-8'
        }
        smtp = smtplib.SMTP_SSL(mailInfo['hostname'])
        smtp.set_debuglevel(1)
        smtp.ehlo(mailInfo['hostname'])
        smtp.login(mailInfo['username'], mailInfo['password'])
        msg = MIMEText(html_code, "html", mailInfo["mailencoding"])
        msg["Subject"] = Header(mailInfo["mailsubject"], mailInfo["mailencoding"])
        msg["from"] = mailInfo["from"]
        msg["to"] = mailInfo["to"]
        smtp.sendmail(mailInfo["from"], mailInfo["to"], msg.as_string())
        smtp.quit()
