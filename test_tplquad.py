import numpy as np
from scipy.integrate import tplquad

lo=-3.
hi=4.
sig0_test=0.8
sig1_test=0.7
vec=np.linspace(lo,hi,50)
x,y,z=np.meshgrid(vec,vec,vec,indexing="ij")

def fcn(x,y,z,sig0,sig1):
    return np.exp(-(z/(2*sig0))**2-((x**2+y**2)/(2*sig1)**2))

integral,error=tplquad(fcn,lo,hi,lo,hi,lo,hi,args=(sig0_test,sig1_test,)) #(func, a, b, gfun, hfun, qfun, rfun, args=(), ...)
print("int(np.exp(-(z/(2*sig0))**2-((x**2+y**2)/(2*sig1)**2))) from -3 to 4 dz dy dx = ",integral,"for sig0,sig1=",sig0_test,sig1_test)

# test with bundled args (the only way that I can anticipate, for now, passing them all the way through the level of P_driver)
# oh shit the only thing I need to do is bundle the args as (arg0,arg1,...,argN,) instead of (arg0,arg1,...,argN) or anything
bundled_args=(sig0_test,sig1_test,)
int2,err2=tplquad(fcn,lo,hi,lo,hi,lo,hi,args=bundled_args)
print("test with bundled args: int2=",int2)