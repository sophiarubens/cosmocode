import numpy as np
from scipy.fft import fftshift,ifftshift, irfftn, set_workers
import time
set_workers(6)

# limiting step in GRF generation is probably not the random draws. probably is the IFFT. 
# include both anyway just to cover my bases / make the MRE more believable

W,X,Y,Z=222,256,256,512

rng=np.random.default_rng()

def serial_ift(w,x,y,z):
    times=np.zeros(w)
    for i in range(w):
        t0=time.time()
        F=rng.normal(size=(x,y,z))
        f=fftshift(irfftn(ifftshift(F),
                          norm="forward",s=(x,y,z)))
        times[i]=time.time()-t0

        if i>0:
            if (i%(w//10)==0):
                print(i/w*100,"pct complete")
    return f,times

def parallel_ift(w,x,y,z):
    t0=time.time()
    F=rng.normal(size=(w,x,y,z))
    f=fftshift(irfftn((ifftshift(F)),
                      norm="forward",s=(x,y,z),axes=(1,2,3)))
    t=time.time()-t0
    return f,t

f_seri,t_seri=serial_ift(W,X,Y,Z)
print("std( serial times) = ",np.std(t_seri))
print("mean(serial times) = ",np.mean(t_seri))

# one call of parallel_ift accomplishes the same thing as one call of  serial_ift
# but, to get similarly robust statistics, I need to call f_para W times and then
# compare the std and mean of the times from the single serial_ift call to the std and mean of parallel_ift/W times
times_para=np.zeros(W)
for i in range(W): 
    f_para,t_para=parallel_ift(W,X,Y,Z)
    times_para[i]=t_para

    if i>0:
        if (i%(W//10)==0): # incur comparable branchy print–induced / collapse postulate slowdown for both branches
                print(i/W*100,"pct complete")
print("std( parallel times / W) = ",np.std(times_para/W))
print("mean(parallel times / W) = ",np.mean(times_para/W))