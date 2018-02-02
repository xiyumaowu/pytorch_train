import json

json_file = open('xianfang.json', 'r')
for line in json_file.readlines():
    house_info = json.loads(line)
    for info in house_info['house_info']:
        print(info)