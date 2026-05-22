import numpy as np
import time
arr1=np.linspace(-3,5)
arr2=np.arange(1,8)
N=500

t_py=np.zeros(N)
for i in range(N):
    t0=time.time()
    res_py=np.sqrt(arr1[None, :]**2 + arr2[:, None]**2)
    t_py[i]=time.time()-t0
print("pythonic times\nmean",np.mean(t_py),"\nstd",np.std(t_py))

t_mg=np.zeros(N)
for i in range(N):
    t0=time.time()
    a1g,a2g=np.meshgrid(arr1,arr2)
    res=np.sqrt(a1g**2+a2g**2)
    t_mg[i]=time.time()-t0
print("meshgrid times\nmean",np.mean(t_mg),"\nstd",np.std(t_mg))