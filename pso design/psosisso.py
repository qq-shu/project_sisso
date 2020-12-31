# Plot particle history as animation
import numpy as np
from sko.PSO import PSO
import time

# define the search formulation demo_func()
def demo_func(x):
    DT, Cr,Ni,C,TT,THT = x
    return -50.58*pow(DT*(Cr+Ni),1/3)-2.24536*C*(TT+DT-THT)-158.7
time_start = time.time()
# define the search range(the main 6 features) which should do PSO search
# here dim means the dimension of features wished to be designed, pop means number of particles, max_iter means the iteration rounds, 
# lb and ub each means the lowest limit and the highest limit of PSO range to each feature
pso = PSO(func=demo_func, dim=6, pop=20, max_iter=400, lb=[30, 0 , 0 , 0 , 0 , 10], ub=[850 , 0.55 , 1.69 , 0.23 , 160 , 30])
pso.record_mode = True
pso.run()
# print out the best target value and the related values of the 6-dimensional designed features
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
time_end = time.time()
# print out the running time of PSO search on SISSO model
print(time_end-time_start)
