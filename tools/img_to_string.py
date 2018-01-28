from PIL import Image
import pytesseract
import os

text = pytesseract.image_to_string(Image.open('test.jpg'), lang='chi_sim')
print(text)