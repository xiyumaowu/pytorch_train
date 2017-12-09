import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)  #reproducibel

EPOCH =1   #训练次数
BATCH_SIZE = 50
LR = 0.001  #学习率， 数据小于1， 越小越高
DOWNLOAD_MNIST = False  #是否需重新下载MNIST数据

train_data = torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), #转换PIL.Image 或者 numpy.naddarray  成 floatTensor（C x H x W) 训练时normalize成（0，0，1）
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = torchvision.datasets.MNIST(root='../data', train=False)
#sharpe from (2000, 28, 28) to (2000,1, 28, 28) in range (0,1)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  #input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1, #input height
                out_channels=16, # n_filters
                kernel_size=5,  #filter size
                stride=1,  #filter movement/step
                padding=2,   #如果想要conv2d 出来的图片长宽没有变化， padding=(kernal_size-1)/2 while stride=1
            ),
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2), #在2x2空间里向下采样， output shape（16，14，14）
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),  #output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  #output shape (32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10)  #fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展开多维的卷积图成（batch_size, 32*7*7
        output = self.out(x)
        return output

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn.forward(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(x), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.data[0]))

test_output = cnn(test_x[:2000])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
# print(pred_y, "prediction number")
# print(test_y[:100].numpy(), "real number")

for i in range(2000):
    if pred_y[i] != test_y[i]:
        print("error number:{}/{}", pred_y[i], test_y[i])

