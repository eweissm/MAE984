from scipy.optimize import minimize
import numpy as np
x0 = np.array(1., 1., 1., 1., 1.)

eqn = lambda x: (x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
bnds = ((-10., 10.), (-10., 10.), (-10., 10.), (-10., 10.), (-10., 10.))

cnst = [{'type': 'eq', 'fun': lambda x: (x[0] + 3. * x[1])},
       {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 2. * x[4]},
       {'type': 'eq', 'fun': lambda x: x[1] - x[4]}]

res = minimize(eqn, x0, method='SLSQP', bounds=bnds, constraints=cnst)

print(res)

