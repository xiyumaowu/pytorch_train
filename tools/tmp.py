import requests
from bs4 import BeautifulSoup

start_url = 'https://xa.fang.lianjia.com/loupan/'
start_html = requests.get(start_url).text
# print(start_html)
total_count = int(BeautifulSoup(start_html, 'lxml').select('.page-box')[0].get('data-total-count'))
if (total_count % 10) == 0 :
    max_page_num = total_count/10
else:
    max_page_num = total_count/10 + 1
print(max_page_num)