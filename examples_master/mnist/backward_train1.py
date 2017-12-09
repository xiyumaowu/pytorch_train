import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) #-1 到 1 的100个数值二元数组
y = x.pow(2) + 0.2*torch.rand(x.size())   # y = x^2

x, y = Variable(x), Variable(y)

#画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hiddens, n_outputs):
        super(Net, self).__init__()  #继续__init__
        self.hidden = torch.nn.Linear(n_features, n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens, n_outputs)

    def forward(self, x): #复写forward函数
        #正向传播输入值， 神经网络分析出输出值
        x = F.relu(self.hidden(x)) #激励函数（隐藏层的线性值）
        x = self.predict(x)  #输出值
        return x


net = Net(n_features=1, n_hiddens=10, n_outputs=1)

# print(net)  #net's structer

optimizer = torch.optim.SGD(net.parameters(), lr=0.5) #传入net的所有参数， 学习率
loss_func = torch.nn.MSELoss() #预测值和真实值的误差计算公式（均方差）

plt.ion()
plt.show()
for t in range(300):
    prediction = net.forward(x)  #喂给net训练数据x， 输出两者的误差
    loss = loss_func(prediction, y)  #计算两者的误差

    optimizer.zero_grad() #清空上一步的残余更新参数值
    loss.backward()  #误差反向传播， 计算参数更新值
    optimizer.step()  #将参数更新值施加到net 的paraeter上

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)