import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from IPython import display

X1 = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
X2 = np.flip(X1, axis=1).copy()
a = np.array(([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]]))
T = 20
p_water = 10 ** (a[0, 0] - a[0, 1] / (T + a[0, 2]))
p_dio = 10 ** (a[1, 0] - a[1, 1] / (T + a[1, 2]))
P = np.array([[28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]])
P = torch.tensor(P, requires_grad=False, dtype=torch.float32)

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

x1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32)
x2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)

a = .0001

for i in range(100):
    P_pred = x1 * torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_water + x2 * torch.exp(
        A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_dio

    loss = (P_pred - P) ** 2
    loss = loss.sum()

    loss.backward()

    with torch.no_grad():
        A -= a * A.grad

        A.grad.zero_()

print('Regression estimation A12 and A21 is:', A)
print('Regression final loss is: ', loss.data.numpy())

P_pred = P_pred.detach().numpy()[0]
P = P.detach().numpy()[0]
x1 = x1.detach().numpy()[0]

plt.plot(x1, P_pred, label='predicted pressure')
plt.plot(x1, P, label='actual pressure')
plt.xlabel('x1')
plt.ylabel('pressure')
plt.legend()
plt.title('comparison between predicted and actual pressure')
plt.show()

# question 2


func = lambda y1, y2: (4 - 2.1 * y1 ** 2 + (y1 ** 4) / 3) * y1 ** 2 + y1 * y2 + (-4 + 4 * y2 ** 2) * y2 ** 2

from hyperopt import hp


# Create the domain space
def para_space():
    space_paras = {'y1': hp.uniform('y1', -3, 3),
                   'y2': hp.uniform('y1', -2, 2)
                   }
    return space_paras


from hyperopt import tpe

# Create the algorithm
tpe_algo = tpe.suggest

from hyperopt import Trials

# Create a trials object
tpe_trials = Trials()

from hyperopt import fmin

# Run 2000 evals with the tpe algorithm
tpe_best = fmin(fn=func, space=para_space,
                algo=tpe_algo, trials=tpe_trials,
                max_evals=2000)

print(tpe_best)
