#! python 3.6, sklear 0.19
import numpy as np
from sklearn import *
# from sklearn.cross_validation import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

print(iris_x[:2,:])
print(iris_y[:10])