import requests

url = 'http://147.128.2.74/?target=&auth_id=&ap_name='
data = {
    'login': 'login',
    'password': 'ossuser',
    'username': 'useross'
}

r = requests.session()
r.post(url=url, data=data)
s = r.get('http://147.128.2.74/?target=&auth_id=&ap_name=')
print(s.text)