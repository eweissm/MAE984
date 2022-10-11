# Import HyperOpt Library
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
