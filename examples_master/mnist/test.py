from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import codecs
from skimage import io

raw_folder = 'raw'
processed_folder = 'processed'
training_file = 'training.pt'
test_file = 'test.pt'
root = '../data'

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)
def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)


# training_set = (
#             read_image_file(os.path.join(root, raw_folder, 'train-images-idx3-ubyte')),
#             read_label_file(os.path.join(root, raw_folder, 'train-labels-idx1-ubyte'))
#         )
test_set = (
            read_image_file(os.path.join(root, raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(root, raw_folder, 't10k-labels-idx1-ubyte'))
        )


def convert_to_img(istrain=True):
    if istrain:
        f = open(root+"/train.txt", 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(training_set[0], training_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f = open(root+"/test.txt", 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            # io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

convert_to_img(False)
# convert_to_img(True)
# with open(os.path.join(root, processed_folder, training_file), 'wb') as f:
#     torch.save(training_set, f)
# with open(os.path.join(root, processed_folder, test_file), 'wb') as f:
#     torch.save(test_set, f)
