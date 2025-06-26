import numpy as np

def ceil05(x):
    print("x=",x,"20*x=",20.*x)
    multceil=np.ceil(20.*x)
    print("multceil=np.ceil(20.*x)=",multceil)
    result=multceil/20.
    print("result=multceil/20.=",result,"\n")
    return result

def floor05(x):
    print("x=",x,"20*x=",20.*x)
    multfloor=np.floor(20.*x)
    print("multfloor=np.floor(20.*x)=",multfloor)
    result=multfloor/20.
    print("result=multfloor/20.=",result,"\n")
    return result

x=                 np.array([1.27,3.31,1.20,5.44,0.99,0.0001,0.06])
xceil05=ceil05(x)
xfloor05=floor05(x)
xceil05_expected=  np.array([1.30,3.35,1.20,5.45,1.00,0.05,  0.10])
xfloor05_expected= np.array([1.25,3.30,1.20,5.40,0.95,0.00,  0.05])

print("np.all(xceil05==xceil05_expected)=",np.all(xceil05==xceil05_expected))
print("np.all(xfloor05==xfloor05_expected)=",np.all(xfloor05==xfloor05_expected))

def ceilflex(x,roundto=0.05):
    print("x=",x,"x/roundto=",x/roundto)
    multceil=np.ceil(x/roundto)
    print("multceil=np.ceil(x/roundto)=",multceil)
    result=multceil*roundto
    print("result=multceil*roundto=",result,"\n")
    return result

def floorflex(x,roundto=0.05):
    print("x=",x,"x/roundto=",x/roundto)
    multfloor=np.floor(x/roundto)
    print("multfloor=np.floor(x/roundto)=",multfloor)
    result=multfloor*roundto
    print("result=multfloor*roundto=",result,"\n")
    return result

xceilflex=ceilflex(x)
xfloorflex=floorflex(x)

print("xceilflex-xceil05_expected=",xceilflex-xceil05_expected)
print("xfloorflex-xfloor05_expected=",xfloorflex-xfloor05_expected)

# print("np.all(xceilflex==xceil05_expected)=",np.all(xceilflex==xceil05_expected))
# print("np.all(xfloorflex==xfloor05_expected)=",np.all(xfloorflex==xfloor05_expected))