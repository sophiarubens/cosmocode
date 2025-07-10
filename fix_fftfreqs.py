import numpy as np
from numpy.fft import fftfreq,fftshift,ifftshift,fftn,irfftn
from power import get_bins

pi=np.pi
twopi=2.*np.pi

Lsurvey=15
Nvox=10
Nk=8

Delta=Lsurvey/Nvox
d3r=Delta**3
kmax_theo=twopi/Delta
kmin_theo=twopi/Lsurvey
print("kmin_theo=",kmin_theo,"\nkmax_theo=",kmax_theo)

T=np.random.normal(loc=0.0, scale=1.0, size=(Nvox,Nvox,Nvox))
T_tilde=fftshift(fftn((ifftshift(T)*d3r)))
modsq_T_tilde= (T_tilde*np.conjugate(T_tilde)).real

k_vec_for_box= twopi*fftshift(fftfreq(Nvox,d=Delta))
kx_grid,ky_grid,kz_grid=np.meshgrid(k_vec_for_box,k_vec_for_box,k_vec_for_box,indexing="ij")
k_grid=np.sqrt(kx_grid**2+ky_grid**2+kz_grid**2)
print(k_vec_for_box)

k_bins,limiting_spacing=get_bins(Nvox,Lsurvey,Nk,"lin") # get_bins(Nvox,Lsurvey,Nk,mode)
print("k_bins=",k_bins)
bin_indices=np.digitize(k_grid,k_bins)

with open("fix_fftfreqs.txt", "w") as f:
    f.write("k_bins=\n"+str(k_bins)+"\n\n")
    f.write("k_vec_for_box=\n"+str(k_vec_for_box)+"\n\n")
    f.write("kgrid=\n")
    for i, slice2d in enumerate(k_grid):
        np.savetxt(f, slice2d, fmt='%6.3f')
        if i < k_grid.shape[0] - 1: 
            f.write('\n')
    f.write("bin_indices=\n")
    for i, slice2d in enumerate(bin_indices):
        np.savetxt(f, slice2d, fmt='%2d')
        if i < k_grid.shape[0] - 1: 
            f.write('\n')

bin_indices_1d=   np.reshape(bin_indices,(Nvox**3,))       # to bin, I use np.bincount, which requires 1D input
print("np.max(bin_indices_1d),np.min(bin_indices_1d)=",np.max(bin_indices_1d),np.min(bin_indices_1d))
modsq_T_tilde_1d= np.reshape(modsq_T_tilde,    (Nvox**3,)) # ^ same preprocessing

# binning
sum_modsq_T_tilde= np.bincount(bin_indices_1d,weights=modsq_T_tilde_1d) # for the ensemble average: sum    of modsq_T_tilde values in each bin # no grid points have the final bin index (-1, Nk-1, whatever you want to call it), so it won't show up in the bincount
N_modsq_T_tilde=   np.bincount(bin_indices_1d) # for the ensemble average: number of modsq_T_tilde values in each bin

# print("BEFORE TRUNCATION: sum_modsq_T_tilde="+np.array2string(sum_modsq_T_tilde, precision=3, suppress_small=True))
# print("BEFORE TRUNCATION: N_modsq_T_tilde="+np.array2string(N_modsq_T_tilde, precision=3, suppress_small=True))
# naïve_quotient=sum_modsq_T_tilde/N_modsq_T_tilde/Lsurvey**3
# print("naïve_quotient=",np.array2string(naïve_quotient, precision=3))

# sum_modsq_T_tilde= sum_modsq_T_tilde[1:-1] # the central voxel has a k below the lowest bin floor, and we won't lose much info by excising it, so focus on the other Nvox**3-1 voxels with k in the bin range (CONFIRMED ON JUN 30TH: N_modsq_T_tilde[0] before pruning is always 1, so my excising intuition seems justified)
# N_modsq_T_tilde=   N_modsq_T_tilde[1:-1]


# print("AFTER TRUNCATION: sum_modsq_T_tilde="+np.array2string(sum_modsq_T_tilde, precision=3, suppress_small=True))
# print("AFTER TRUNCATION: N_modsq_T_tilde="+np.array2string(N_modsq_T_tilde, precision=3, suppress_small=True))

# avg_modsq_T_tilde= np.zeros(Nk)
# nonemptybins=np.nonzero(N_modsq_T_tilde)
# avg_modsq_T_tilde[nonemptybins]=sum_modsq_T_tilde[nonemptybins]/N_modsq_T_tilde[nonemptybins]

# print("ensemble average quotient = ",np.array2string(sum_modsq_T_tilde[nonemptybins]/N_modsq_T_tilde[nonemptybins]/Lsurvey**3, precision=3))

avg_modsq_T_tilde=sum_modsq_T_tilde/N_modsq_T_tilde
P=np.array(avg_modsq_T_tilde/Lsurvey**3)

# P=P[:-1] # actually ignore the first shell of corner points (I was using a k-value outside the largest box-enclosed sphere as a bin floor, even when, in theory, I agree that it's probably okay to ignore the not-great stats in the corners of a box) 
k=k_bins[:-1]

print("AFTER TRUNCATION 2: P="+np.array2string(P, precision=3, suppress_small=True))
print("AFTER TRUNCATION 2: k="+np.array2string(k, precision=3, suppress_small=True))