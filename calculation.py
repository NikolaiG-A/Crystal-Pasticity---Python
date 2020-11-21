import numpy as np
from scipy.optimize import minimize
from scipy import optimize
import matplotlib.pyplot as plt
from functions import *
from classes import *
import time
nx,ny=(3,3)
h0=1
x = np.linspace(0,(nx-1)*h0,nx)
y = np.linspace(0,(ny-1)*h0,ny)

X,Y = np.meshgrid(x,y)

p = np.concatenate((X.reshape((nx*ny,1)),Y.reshape((nx*ny,1))),axis=1)
T = np.empty((0,3), int)
for j in range(ny-1):
    for i in range(nx-1):
        T = np.append(T,[[i+j*ny,(i+1)+j*ny,i+(j+1)*nx],\
                          [(i+1)+(j+1)*ny,i+(j+1)*ny,(i+1)+j*nx]],axis=0)
geom=geom_gen(p,T)


en_par=en_parameters(en_type='poly', K=1, C='sq')
ds=0.001
s_max=0.01
s_min=0
s=np.arange(s_min,s_max,ds)

U=np.zeros((2*p.shape[0],len(s)))

print(np.eye(2)+def_grad_e(p,U,T[0,:]).F)
for i in range(len(s)):
    BCs_val,bulk_val = bound_cond(geom,bc_type='F',s=s[i],u0=None,v0=None,bulk_load=None)
    fun_opt = lambda W: global_energy_constr(en_par,p,W,T,BCs_val,bulk_val)
    if i==0:
        W0=np.zeros(bulk_val.free.shape[0])+0.02*ds*np.random.rand(bulk_val.free.shape[0])-0.01*ds
    else:
        W0=W+0.02*ds*np.random.rand(bulk_val.free.shape[0])-0.01*ds
    time_start = time.time()
    res = minimize(fun_opt, W0, method='L-BFGS-B', jac=True, options={'gtol': 1e-09})
    time_fin = time.time()
    W=res.x
    dW=res.jac
    U[:,i]=total_displ(p,W,T,BCs_val,bulk_val) 
    print('Finished loading step ',str(s[i]),', maximum load ',str(s_max),', norm grad', str(np.linalg.norm(dW)))
    print(time_fin-time_start)
    
plt.figure(0)

plt.plot(p[:,0]+U[::2,0], p[:,1]+U[1::2,0], 'ro')

plt.figure(1)
plt.plot(p[:,0]+U[::2,5], p[:,1]+U[1::2,5], 'bo')

plt.figure(2)
plt.plot(p[:,0]+U[::2,-1], p[:,1]+U[1::2,-1], 'ko')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(p[:,0], p[:,1], 'ro') # Returns a tuple of line objects, thus the comma