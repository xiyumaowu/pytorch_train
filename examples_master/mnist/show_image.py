import numpy as np
import struct
import matplotlib.pyplot as plt

untrained_image_path = "../data/raw/train-images-idx3-ubyte"
trained_image_path = "../data/processed/training.pt"
binfile = open(untrained_image_path, 'rb')
buf = binfile.read()     #读取bin文件

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')  #'>IIII'是说使用大端法读取4个unsinged int32

for i in range(10):
    im = struct.unpack_from('>784B', buf, index)  #读取一个28*28的图片
    index += struct.calcsize('>784B')

    im = np.array(im)
    im = im.reshape(28, 28)  #转换成一个28*28的矩阵

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.show()