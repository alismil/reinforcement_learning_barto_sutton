"""Exercise 11.3"""

import numpy as np
from random import random
import matplotlib.pyplot as plt
import torch

S = torch.tensor([
    [2,0,0,0,0,0,0,1],
    [0,2,0,0,0,0,0,1],
    [0,0,2,0,0,0,0,1],
    [0,0,0,2,0,0,0,1],
    [0,0,0,0,2,0,0,1],
    [0,0,0,0,0,2,0,1],
    [0,0,0,0,0,0,1,2]
], requires_grad=False, dtype=torch.float64)

alpha = 0.01
gamma = 0.99
w = torch.ones(8, dtype=torch.float64)
w[-2] = 10

def get_action_feature(state, action):
    # for any current state, dashed actions always take us to the first 6 states, 
    # so we can consider their features to be the average features of the first 6 states
    if action == 'dashed':
        return S[:-1, :].mean(dim=0)
    # the solid action always takes us to the last state, hence we use its features to represent it
    return S[-1]

def q(state, action, w):
    action_feature = get_action_feature(state, action)
    return torch.inner(action_feature, w).item()

def max_q(state, w):
    return max(q(state, 'dashed', w), q(state, 'solid', w))

def choose_action():
    # the same for any state
    if random() > 1/7:
        return 'dashed'
    return 'solid'

def get_state_and_reward_from_action(action):
    # all rewards are 0
    # dashed actions take us to any of the first 6 states with equal probability
    if action == 'dashed':
        return [0, S[np.random.randint(6)]]
    # solid action takes us to the last state
    return [0, S[-1]]

weights = []

# since there is no specific terminal state mentioned in the example, we can treat 
# this as a continuous task or a sequence of steps in one episonde
S1 = S[np.random.randint(7)]
for i in range(1000):
    A = choose_action()
    R, S2 = get_state_and_reward_from_action(A)
    # currently hard coded to dq/dw = x, change to torch.grad
    w += S1*alpha*(R + gamma*max_q(S2,w) - q(S1,A,w))
    weights.append(w.clone())
    S1=S2

# plot the output
x = np.linspace(0,1000,num=1000)
all_weights = torch.stack(weights,0)
y0 = all_weights[:,0]
y1 = all_weights[:,1]
y2 = all_weights[:,2]
y3 = all_weights[:,3]
y4 = all_weights[:,4]
y5 = all_weights[:,5]
y6 = all_weights[:,6]
y7 = all_weights[:,7]
plt.plot(x, y0, label = "x0")
plt.plot(x, y1, label = "x1")
plt.plot(x, y2, label = "x2")
plt.plot(x, y3, label = "x3")
plt.plot(x, y4, label = "x4")
plt.plot(x, y5, label = "x5")
plt.plot(x, y6, label = "x6")
plt.plot(x, y7, label = "x7")
plt.legend()
plt.show()
