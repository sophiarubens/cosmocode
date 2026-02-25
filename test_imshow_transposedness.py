import numpy as np
from matplotlib import pyplot as plt

a=14
b=6

arr=np.zeros((a,b))
arr[3,:]=3
arr[5,:]=5
arr[:,4]=4

plt.figure()
plt.imshow(arr,origin="lower")
plt.colorbar()
plt.savefig("imshow_transposedness.png")