import numpy as np

a=np.random.randn(4,4)
print("original a:\n",a,"\n\n")
a[1:3,1:3]=0
print("zero out the middle chunk:\n",a,"\n\n")
# a[np.nonzero(a==0)]=10 # I know this works but it looks ridiculous
a[a==0]=10
print("overwrite the zeroed-out chunk:\n",a,"\n\n")