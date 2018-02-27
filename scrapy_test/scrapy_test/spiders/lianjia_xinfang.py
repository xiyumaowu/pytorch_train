# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup
import re
import  requests
import sys
from email.mime.text import MIMEText
import smtplib
from email.header import Header

class Lianjia_xinfangSpider(scrapy.Spider):
    name = 'lianjia_xinfang'
    allowed_domains = ['xa.lianjia.com']
    start_url = 'https://xa.fang.lianjia.com/loupan/'
    start_html = requests.get(start_url).text
    start_urls = []
    total_count = int(BeautifulSoup(start_html, 'lxml').select('.page-box')[0].get('data-total-count'))
    if total_count % 10 == 0 :
        max_page_num = int(total_count/10)
    else:
        max_page_num = int(total_count/10) + 1
    bashurl = 'https://xa.fang.lianjia.com/loupan/pg{}'
    for i in range(1, max_page_num+1):
        start_urls.append(bashurl.format(i)) # get start urls
    start_urls = ['https://xa.fang.lianjia.com/loupan/pg1']

    def __init__(self):
        self.filter_count = 0
        self.round_num = 0
        self.html_code = ''
        try:
            with open('xinfang.csv', 'a+') as self.csv:
                csv_tilte = '编号,' + '标题,' + '详细信息,' + '位置信息,' + '关注,' + '总价,' + '单价,' + '链接\n'
                self.csv.write(csv_tilte)
        except Exception as e:
            print(e)
            sys.exit()


    def parse(self, response):
        self.round_num += 1
        resblocks = BeautifulSoup(response.text, 'lxml').select('.resblock-desc-wrapper')
        for resblock in resblocks:
            self.name = resblock.select('.resblock-name')[0].select('.name')[0].text
            self.location = resblock.select('.resblock-location')[0].text.replace('\n', '')
            self.room = resblock.select('.resblock-room')[0].text.replace('\n', '')
            self.area = resblock.select('.resblock-area')[0].text.replace('\n', '')
            self.unitprice = resblock.select('.resblock-price')[0].text.replace('\n', '')
            print(self.name, self.location,self.room, self.area, self.unitprice)


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
