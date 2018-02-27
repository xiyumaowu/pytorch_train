import xml.etree.cElementTree as et
import os

try:
    tree = et.parse('test1.xml')
    root = tree.getroot()
    child1 = root.getchildren()
    print(child1)
except Exception as e:
    print(e)