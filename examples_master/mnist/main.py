from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(        #加载训练集
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([     #创建transform来做图像预处理
                      transforms.ToTensor(),          #把PIL.Image(RGB) 或者numpy.ndarray(H x W x C) 从0到255的值映射到0到1的范围内，并转化成Tensor格式
                       transforms.Normalize((0.1307,), (0.3081,))   #此转换类作用于torch.*Tensor。给定均值(R, G, B)和标准差(R, G, B)，用公式channel = (channel - mean) / std进行规范化。
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   #二维卷积层, 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）
        '''
        in_channels(int) – 输入信号的通道
        out_channels(int) – 卷积产生的通道
        kerner_size(int or tuple) - 卷积核的尺寸
        stride(int or tuple, optional) - 卷积步长
        padding(int or tuple, optional) - 输入的每一条边补充0的层数
        dilation(int or tuple, optional) – 卷积核元素之间的间距
        groups(int, optional) – 从输入通道到输出通道的阻塞连接数
        bias(bool, optional) - 如果bias=True，添加偏置
      '''
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        '''如果特征图中相邻像素是强相关的（在前几层卷积层很常见），那么iid dropout不会归一化激活，而只会降低学习率。
        在这种情形，nn.Dropout2d()可以提高特征图之间的独立程度，所以应该使用它。
        p(float, optional) - 将元素置0的概率。
        in-place(bool, optional) - 若设置为True，会在原地执行操作。
        输入： (N,C,H,W)
        输出： (N,C,H,W)
        （与输入形状相同）'''
        self.fc1 = nn.Linear(320, 50)  #对输入数据做线性变换：y=Ax+b
        '''
        in_features - 每个输入样本的大小
        out_features - 每个输出样本的大小
        bias - 若设置为False，这层不会学习偏置。默认值：True
        '''
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  #max_pool2d: 在kt x kh x kw区域中应用步长为dt x dh x dw的二维平均池化操作。输出特征的数量等于 input planes / dt
       #relu: 非线性激活函数
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()  #将module设置为 training mode,仅仅当模型中有Dropout和BatchNorm是才会有影响
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/20 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()  #将模型设置成evaluation模式,仅仅当模型中有Dropout和BatchNorm是才会有影响。
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



for epoch in range(1, args.epochs + 11):
    print("%d of 10" %epoch )
    train(epoch)
    test()
