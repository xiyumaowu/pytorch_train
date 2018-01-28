import re

a = 'asd3424fas 23.423klj;ljk'
b = '7681元'
re_set = re.compile(r'\d+.?\d+')
re_get = re.findall(re_set, a)
print(re_get)