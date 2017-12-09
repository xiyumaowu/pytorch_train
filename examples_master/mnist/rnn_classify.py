import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision

EPOCH = 1 # trainning times
BATCH_SIZE = 64
LR = 0.01  # 学习率
TIME_STEP = 28
INPUT_SIZE = 28

train_data = torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=torchvision.transforms.ToTensor(), #转换PIL.Image 或者 numpy.naddarray  成 floatTensor（C x H x W) 训练时normalize成（0，0，1）
    download=False,
)

# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i'%train_data.train_labels[0])
# plt.show()


train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = torchvision.datasets.MNIST(root='../data', train=False, transform=torchvision.transforms.ToTensor())
#sharpe from (2000, 28, 28) to (2000,1, 28, 28) in range (0,1)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(  #也可以用nn.RNN, 但LSTM效果好得多
            input_size=28,  #图片第行的数据像素点
            hidden_size=64,  #rnn hidder unit
            num_layers=1, # 有几层rnn layers
            batch_first=True # input, output是否以batch size为第一特征集 （batch_sixe, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)  #输出层

    def forward(self, x):
        r_o, (h_n, h_c) = self.rnn(x,None)  #None 表示 Hidder state会为全0的state
        out = self.out(r_o[:,-1, :])  #选取最后一个time_step的值, 这是r_o[:,-1,:] 也的h_n的值
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# for epoch in range(1):
#     for step, (x,y) in enumerate(train_loader):
#         b_x = Variable(x.view(-1, 28, 28)) # shape x to (batch, time_step, input_size)
#         b_y = Variable(y)
#
#         output = rnn(b_x)
#         loss = loss_func(output, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % 50 == 0:
#             test_output = rnn(test_x.view(-1, 28, 28))
#             pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
#             accuracy = sum(pred_y == test_y) / float(test_y.size)
#             print("Epoch:", epoch, "| train loss: %.4f" % loss.data[0], "| Accuracyy: %.2f" % accuracy)
#
# torch.save(rnn, "../data/rnn_classify.pkl")
rnn2 = torch.load("../data/rnn_classify.pkl")

for epoch in range(1):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28)) # shape x to (batch, time_step, input_size)
        b_y = Variable(y)

        output = rnn2(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn2(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print("Epoch:", epoch, "| train loss: %.4f" % loss.data[0], "| Accuracyy: %.2f" % accuracy)

test_output = rnn2(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, "prediction number")
print(test_y[:10], "real number")

# error_count = 0
# for i in range(2000):
#     if pred_y[i] != test_y[i]:
#         print("error number:{}/{}", pred_y[i], test_y[i])
#         error_count += 1
# print("error rate: {.6f}", error_count/2000)
#
