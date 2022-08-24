# import lib

from scipy import optimize

# initial guess
x0 = [1, 1, 1, 1, 1]
fun = lambda x: (x[0]-x[1])**2+(x[1]+x[2]-2)**2+(x[3]-1)**2+(x[4]-1)**2

res = minimize(fun, x0, bounds=bnds, constraints=cnst)
