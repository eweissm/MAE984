

from scipy.optimize import minimize
import numpy as np

# question 1
## initial conditions all = 1
print('all ICs = 1:')
x0 = np.array([1., 1., 1., 1., 1.])

eqn = lambda x: (x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
bnds = ((-10., 10.), (-10., 10.), (-10., 10.), (-10., 10.), (-10., 10.))

cnst = [{'type': 'eq', 'fun': lambda x: (x[0] + 3. * x[1])},
       {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 2. * x[4]},
       {'type': 'eq', 'fun': lambda x: x[1] - x[4]}]

res = minimize(eqn, x0, method='SLSQP', bounds=bnds, constraints=cnst)

xOpt = res.x
print(xOpt)
print('contraint 1 = ')
print(xOpt[0] + 3. * xOpt[1])
print('contraint 2 = ')
print(xOpt[2] + xOpt[3] - 2. * xOpt[4])
print('contraint 3 = ')
print(xOpt[1] - xOpt[4])
print('function = ')
print((xOpt[0] - xOpt[1]) ** 2 + (xOpt[1] + xOpt[2] - 2) ** 2 + (xOpt[3] - 1) ** 2 + (xOpt[4] - 1) ** 2)

#### initial conditions all = 5
print('all ICs = 5:')
x0 = np.array([5., 5., 5., 5., 5.])

eqn = lambda x: (x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
bnds = ((-10., 10.), (-10., 10.), (-10., 10.), (-10., 10.), (-10., 10.))

cnst = [{'type': 'eq', 'fun': lambda x: (x[0] + 3. * x[1])},
       {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 2. * x[4]},
       {'type': 'eq', 'fun': lambda x: x[1] - x[4]}]

res = minimize(eqn, x0, method='SLSQP', bounds=bnds, constraints=cnst)

xOpt = res.x
print(xOpt)
print('contraint 1 = ')
print(xOpt[0] + 3. * xOpt[1])
print('contraint 2 = ')
print(xOpt[2] + xOpt[3] - 2. * xOpt[4])
print('contraint 3 = ')
print(xOpt[1] - xOpt[4])
print('function = ')
print((xOpt[0] - xOpt[1]) ** 2 + (xOpt[1] + xOpt[2] - 2) ** 2 + (xOpt[3] - 1) ** 2 + (xOpt[4] - 1) ** 2)