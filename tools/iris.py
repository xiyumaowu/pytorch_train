from urllib import request

# url = 'http://aima.cs.berkeley.edu/data/iris.csv'
# header = {"User-Agent" : "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"}
# req = request.Request(url=url, headers=header)
# u = request.urlopen(req)
#
# localfile = open('iris.csv', 'w')
# localfile.write(bytes.decode(u.read()))
# localfile.close()


from numpy import genfromtxt, zeros
#read the first 4 columns
data = genfromtxt('iris.csv', delimiter=',', usecols=(0,1,2,3))
#read the fifth column, typeof flower
target = genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)

# print(data.shape)
# print(target.shape)
# print(set(target))
#bulid a collection of unique elements
# set(['setosa', 'versicolor', 'virginica'])


# from pylab import plot, show
# plot(data[target=='setosa',0], data[target=='setosa',2],'bo')
# plot(data[target=='versicolor',0],data[target=='versicolor',2], 'ro')
# plot(data[target=='virginica', 0], data[target=='virginica', 2], 'go')
# show()

#data[:,0], 二组数组中第一列 [:1], 第二列

# from pylab import figure, subplot, hist, xlim, show
#
# xmin = min(data[:,0])
# xmax = max(data[:,0])
# figure()
# subplot(411)
# hist(data[target=='setosa', 0], color='b', alpha=.7)
# xlim(xmin, xmax)
# subplot(412)
# hist(data[target=='versicolor',0], color='r', alpha=.7)
# xlim(xmin, xmax)
# subplot(413)
# hist(data[target=='virginica',0], color='g', alpha=.7)
# xlim(xmin, xmax)
# subplot(414)
# hist(data[:,0], color='y', alpha=.7)
# xlim(xmin, xmax)
# show()

t = zeros(len(target))
t[target=='setosa'] = 1
t[target=='versicolor'] =2
t[target=='virginica'] =3

from sklearn.naive_bayes import  GaussianNB
clssifier = GaussianNB()
clssifier.fit(data, t)  # training on the iris dataset
# print(clssifier.predict(data))

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data,t, test_size=0.4, random_state=0)
clssifier.fit(train, t_train)
print(clssifier.score(test, t_test))