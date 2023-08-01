"""
Exercise 11.3
If we follow policy b as outlined in the book, then given we are in any state s, 
we are equally likely to end up in any state s', i.e. p[s'|s] = 1/7 for all s' and s.
"""

import numpy as np
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

def q(f, w):
    return torch.inner(f, w)

def max_q(w):
    values = [torch.inner(s, w) for s in S]
    return max(values)

weights = []

for i in range(1000):
    S1 = S[np.random.randint(7)]
    # since reward is 0 for all states and actions
    w += S1*alpha*(gamma*max_q(w) - q(S1,w))
    weights.append(w.clone())

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
