import re
import requests
from bs4 import BeautifulSoup


def generate_allurl(user_in_num=3):
    url = "http://xa.sofang.com/esfsale/area/bl{}/"
    for url_next in range(1, int(user_in_num)):
        print(url.format(url_next))
        yield url.format(url_next)

def get_allurl(generate_allurl):
    for i in generate_allurl(3):
        print(i)
    # get_url = requests.get(generate_allurl)
    # print(get_url)
    # if get_url.status_code == 200:
    #     soup = BeautifulSoup(get_url.text, 'lxml')
    #     house_msg = soup.select('.house_msg')
    #     print(house_msg)
        # re_set = re.compile('<li.*?class="clear">.*?<a.*?class="img.*?".*?href="(.*?)"')
        # re_get = re.findall(re_set, get_url.text)
        # return re_get

def main():
    get_allurl(5)

if __name__ == "__main__":
    main()