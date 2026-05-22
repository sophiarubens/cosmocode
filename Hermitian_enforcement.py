from matplotlib import pyplot as plt
import numpy as np

# def dagger(M):
#     return np.conj(M.T)

# def Hermitianize(M):
#     lgsq=np.min(M.shape)
#     largest_square_sub_M=M[:lgsq,:lgsq]
#     M_dagger=dagger(largest_square_sub_M)
#     M_Herm_part=0.5*(M[:lgsq,:lgsq]+M_dagger[:lgsq,:lgsq])
#     M_Herm=np.copy(M)
#     M_Herm[:lgsq,:lgsq]=M_Herm_part
#     return M_Herm

def dagger(M):
    return np.conj(M.T)
def Hermitianize(M): # wrong because the largest cube might be larger than a half-axis
    M_shape=M.shape
    lgsq=np.min(M_shape)
    chunk=tuple(slice(0,lgsq) for _ in M_shape)
    largest_square_sub_M=M[chunk]
    M_dagger=dagger(largest_square_sub_M)
    M_Herm_part=0.5*(M[chunk]+M_dagger[chunk])
    M_Herm=np.copy(M)
    M_Herm[chunk]=M_Herm_part
    return M_Herm
# def Hermitianize(M): # wrong because the half-axes probably do not specify a cube
#     M_shape=M.shape
#     for ax,n in enumerate(M_shape):
#         n2=int(n//2) # half-axis extent. safe for the DFT even-voxel convention
#         chunk=tuple(slice(None) for _ in M_shape)
#         chunk[ax]=slice(0,n2)

#         # no!!!
#         M_dagger
#         M_Herm_part=0.5*(M[chunk]+M_dagger[chunk])
#         M[chunk]
# def Hermitianize(M):
#     M_shape=M.shape
#     supercube_dim=np.max(M_shape)
#     M_padded=np.zeros((supercube_dim,supercube_dim,supercube_dim),dtype=np.complex128)
#     M_location=tuple(slice(0,l) for l in M_shape)
#     M_padded[M_location]=M # should be okay because M is corner-origin
#     M_padded_dagger=dagger(M_padded)
#     M_super_Herm=0.5*(M_padded+M_padded_dagger)
#     return M_super_Herm[M_location]

a=np.asarray([[ 1,   2],  [ 3,    4  ]            ])
b=np.asarray([[-9.8,-7.6],[-5.4, -3.2]            ])
c=np.asarray([[ 2.3, 4.5],[ 1.8, -2.4],[ 0.1, -3.4]])
d=np.asarray([[-5.5, 0. ],[ 0.2, -1.3],[-2.1,  1.1]])
M=a+1j*b
M_dagger=dagger(M)
M_Herm=Hermitianize(M)

print(M)
print(M_dagger)
print(M_Herm)
print(M_Herm==dagger(M_Herm))

N=c+1j*d
N_dagger=dagger(N)
N_Herm=Hermitianize(N)
print(N_dagger)
print(N_Herm)
print(N_Herm[:2,:2]==dagger(N_Herm[:2,:2]))