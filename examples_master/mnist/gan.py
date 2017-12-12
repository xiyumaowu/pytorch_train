'''新手画家 (Generator) 在作画的时候需要有一些灵感 (random noise),
我们这些灵感的个数定义为 N_IDEAS. 而一幅画需要有一些规格, 我们将这幅画的画笔数定义一下,
 N_COMPONENTS 就是一条一元二次曲线(这幅画画)上的点个数. 为了进行批训练,
 我们将一整批话的点都规定一下(PAINT_POINTS).'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)
np.random.seed(1)

BATCH_SIZE = 64
LR_G = 0.0001  #learning rate for generator
LR_D = 0.0001  #learning rate for discriminator
N_INDEAS = 5   #
ART_COMPONENTS = 15  # total points G can draw in the canvers
PAINT_POINTS = np.stack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works():
    a  = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paints = a*np.power(PAINT_POINTS, 2) + (a-1)
    paints = torch.from_numpy(paints).float()
    return Variable(paints)

G = nn.Sequential(
    nn.Linear(N_INDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
plt.ion()
for step in range(10000):
    artist_paintings = artist_works()  # real paints from artist
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_INDEAS))  # random ideas
    G_paints = G(G_ideas)

    prob_artist0 = D(artist_paintings)  #D try to increas this prob
    prob_artist1 = D(G_paints)     # D try to reduce this prob

    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_variables=True)
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paints.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()





