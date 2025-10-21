import numpy as np

test_arr=np.array([[1,-3,4,5,5,-8,5],[2,2,0,-1,-1,7,-1]]).T
want=[5,-1]

keep=np.nonzero(test_arr==want)
print("keep=",keep)
print("test_arr[keep]=",test_arr[keep])
# print("test_arr[1][keep[1]]=",test_arr[1][keep[1]])