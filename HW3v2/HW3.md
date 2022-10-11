## Question 1
 We want to minimize the distance between the our prediction and the known values. This is done with a normal regression.
 
 $$ min_{A12, A21} \sum_{n=1} ^{11}\left(P\left(Xi,A\right)-P_{given}\right)^2$$
 
 Then this is solved using a gradient decent algorithm as is seen in the code. This give us the solution that:
Regression estimation A12 and A21 is: 1.9110, 1.7293, with the regression final loss at:  0.87836564

We can see this  answer is quite acurate when compared to the real data, as shown in the graph

![image](https://user-images.githubusercontent.com/73143081/194738405-90d88e00-947f-4cb5-97f8-6ed15ee88cc3.png)

The code for this question can be found below
```
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
```

## Question 2

The code for this section can be found in main.py and/ or Question2.py. For this question I used the hyperopt package, which found that to minimize the function, the parameters should be:

x1= 0.09117672093095912, and x2 = -0.7282924944664971

This was found with the following code:

```
from hyperopt import tpe, hp, fmin

def objective(params):
    x, y = params['x'], params['y']
    return (4-(2.1*x**2)+((x**4)/3))*x**2+(x*y)+(-4+(4*y**2))*y**2


# Define the search space of x between -10 and 10.
space = {
    'x': hp.uniform('x', -3, 3),
    'y': hp.uniform('y', -2, 2)
}
best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm
    max_evals=1000 # Number of optimization attempts
)
print(best)
```
