import numpy as np
from matplotlib import pyplot as plt
from power_class import *
from bias_helper_fcns import *
import time
# import powerbox

Lsurvey=126 # 63
Nvox=52 # 10 
Nk = 11 # 8
mode="lin"
Nrealiz=50

t0=time.time()
colours=plt.cm.Blues(np.linspace(0.2,1,Nrealiz))

sigma02=1e3 # wide   in config space
sigma12=10  # medium in config space
sigma22=0.5 # narrow in config space
beamfwhm_x=3.5 # kind of hacky but enough to stop it from entering the Delta-like case every single time (for the numerics as of 2025.07.07 09:47, the voxel scale comparison value is ~2.42)
beamfwhm_y=np.copy(beamfwhm_x)
sigLoS0=np.sqrt(2.*sigma02)
sigLoS1=np.sqrt(2.*sigma12)
sigLoS2=np.sqrt(2.*sigma22)
r00=np.sqrt(np.log(2))*sigLoS0
r10=np.sqrt(np.log(2))*sigLoS1
r20=np.sqrt(np.log(2))*sigLoS2
print("sigLoS0,sigLoS1,sigLoS2,r00,r10,r20=",sigLoS0,sigLoS1,sigLoS2,r00,r10,r20)
bundled0=(sigLoS0,beamfwhm_x,beamfwhm_y,r00,)
bundled1=(sigLoS1,beamfwhm_x,beamfwhm_y,r10)
bundled2=(sigLoS2,beamfwhm_x,beamfwhm_y,r20,)

# class cosmo_stats(object):
#     def __init__(self,
#                  Lsurvey,                                                                # nonnegotiable for box->spec and spec->box calcs
#                  T=None,P_fid=None,Nvox=None,                                            # need one of either T or P to get started; I also check for any conflicts with Nvox
#                  primary_beam=None,primary_beam_args=None,primary_beam_type="Gaussian",  # primary beam considerations
#                  Nk0=10,Nk1=0,binning_mode="lin",                                        # binning considerations for power spec realizations
#                  realization_ceiling=50,frac_tol=0.05,                                   # max number of realizations
#                  k0bins_interp=None,k1bins_interp=None,                                  # bins where it would be nice to know about P_converged
#                  P_realizations=None,P_converged=None                                    # power spectra related to averaging over those from dif box realizations
#                  ):  

# idx=-0.9 # DECAYING   power law
idx=2.3  # INCREASING power law
# idx=0 # flat power spec / white noise box
ktest=np.linspace(twopi/Lsurvey,twopi*Nvox/Lsurvey,Nk)
Ptest=ktest**idx

unmodulated=cosmo_stats(Lsurvey,
                        P_fid=Ptest,Nvox=Nvox,
                        Nk0=Nk)
modulated_0=cosmo_stats(Lsurvey,
                        P_fid=Ptest,Nvox=Nvox,
                        primary_beam=custom_response,primary_beam_args=bundled0,
                        Nk0=Nk)
modulated_1=cosmo_stats(Lsurvey,
                        P_fid=Ptest,Nvox=Nvox,
                        primary_beam=custom_response,primary_beam_args=bundled1,
                        Nk0=Nk)
modulated_2=cosmo_stats(Lsurvey,
                        P_fid=Ptest,Nvox=Nvox,
                        primary_beam=custom_response,primary_beam_args=bundled2,
                        Nk0=Nk)
# assert(1==0), "does it even make it through the __init__?"

print("unmod")
unmodulated.generate_box()
print("mod 0")
modulated_0.generate_box()
print("mod 1")
modulated_1.generate_box()
print("mod 2")
modulated_2.generate_box()
# fig,axs=plt.subplots(1,3)
# axs[0].imshow(modulated_0.T[:,:,3])
# axs[0].set_title("modulated_0.T[:,:,3]")
# axs[1].imshow(modulated_0.T[:,37,:])
# axs[1].set_title("modulated_0.T[:,37,:]")
# axs[2].imshow(modulated_0.T[18,:,:])
# axs[2].set_title("modulated_0.T[18,:,:]")
# plt.tight_layout()
# plt.savefig("mod0_box_slices.png")
# plt.show()
# assert(1==0), "can I generate a single box?"

print("unmod")
unmodulated.generate_P() # this is a reconstruction test
plt.figure()
# plt.plot(unmodulated.k0bins,unmodulated.P_realizations[0],label="reconstructed") # would need to multiply unmodulated.k0bins by 2 for the curves to line up (I checked) (but, where is this coming from?)
plt.plot(ktest,unmodulated.P_realizations[0],label="reconstructed")
plt.plot(ktest,Ptest,label="fiducial")
plt.legend()
plt.xlabel("k")
plt.ylabel("P")
plt.savefig("test_reconstr_unmod_class.png")
plt.show()
# assert (1==0), "can I generate a single power spec with an identity primary beam?" # (if the cosmo_stats object is not initialized with a box, as it is not here, a box will be generated before a power spec is calcualted)

print("mod 0:")
modulated_0.generate_P()
print("mod 1:")
modulated_1.generate_P()
print("mod 2:")
modulated_2.generate_P()
fig,axs=plt.subplots(1,3)
axs[0].plot(ktest,modulated_0.P_realizations[0])
axs[1].plot(ktest,modulated_1.P_realizations[0])
axs[2].plot(ktest,modulated_2.P_realizations[0],label="reconstructed")
labels=["","","fiducial"]
for i in range(3):
    axs[i].plot(ktest,Ptest,label=labels[i])
    axs[i].set_xlabel("k")
    axs[i].set_ylabel("P")
plt.legend()
fig.tight_layout()
plt.savefig("test_reconstr_class.png")
plt.show()
assert(1==0), "can I generate a single power spec with a non-identity primary beam?"

modulated_1.avg_realizations()
assert(1==0), "can I average over realizations? (requires generating boxes and power spectra and checking convergence)"


assert(1==0), "can I interpolate a converged power spec?" # could also test re-test interpolation independently by interpolating a P_fid or a single realization
assert(1==0), "can I generalize these tests to a cylindrically binned power spec?"