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
maxvals=0.
maxvals_mod0=0.
maxvals_mod1=0.
maxvals_mod2=0.
colours=plt.cm.Blues(np.linspace(0.2,1,Nrealiz))

vec=Lsurvey*np.fft.fftshift(np.fft.fftfreq(Nvox))
xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
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
modulation0=custom_response(xgrid,ygrid,zgrid,sigLoS0,beamfwhm_x,beamfwhm_y,r00)
bundled1=(sigLoS1,beamfwhm_x,beamfwhm_y,r10)
modulation1=custom_response(xgrid,ygrid,zgrid,sigLoS1,beamfwhm_x,beamfwhm_y,r10)
bundled2=(sigLoS2,beamfwhm_x,beamfwhm_y,r20,)
modulation2=custom_response(xgrid,ygrid,zgrid,sigLoS2,beamfwhm_x,beamfwhm_y,r20)
allvals= np.zeros((Nk,Nrealiz))
allvals0=np.zeros((Nk,Nrealiz))
allvals1=np.zeros((Nk,Nrealiz))
allvals2=np.zeros((Nk,Nrealiz))

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

idx=-0.9 # DECAYING   power law
# idx=2.3  # INCREASING power law
# idx=0 # flat power spec / white noise box
ktest=np.linspace(twopi/Lsurvey,twopi*Nvox/Lsurvey,Nk)
Ptest=ktest**idx

unmodulated=cosmo_stats(Lsurvey,
                        P_fid=Ptest,Nvox=Nvox,
                        Nk0=Nk)
modulated_0=cosmo_stats(Lsurvey,
                        T=None,P_fid=Ptest,Nvox=Nvox,
                        primary_beam=custom_response,primary_beam_args=bundled0,
                        Nk0=Nk)
modulated_1=cosmo_stats(Lsurvey,
                        T=None,P_fid=Ptest,Nvox=Nvox,
                        primary_beam=custom_response,primary_beam_args=bundled1,
                        Nk0=Nk)
modulated_2=cosmo_stats(Lsurvey,
                        T=None,P_fid=Ptest,Nvox=Nvox,
                        primary_beam=custom_response,primary_beam_args=bundled2,
                        Nk0=Nk)
# assert(1==0), "does it even make it through the __init__?"

unmodulated.generate_box()
modulated_0.generate_box()
modulated_1.generate_box()
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

unmodulated.generate_P() # this is a reconstruction test
plt.figure()
plt.plot(unmodulated.k0bins,unmodulated.P_realizations[0])
plt.savefig("test_reconstr_unmod_class.png")
plt.show()
assert (1==0), "can I generate a single power spec with an identity primary beam?" # (if the cosmo_stats object is not initialized with a box, as it is not here, a box will be generated before a power spec is calcualted)

modulated_0.generate_P()
modulated_1.generate_P()
modulated_2.generate_P()
fig,axs=plt.subplots(1,3)
axs[0].plot(modulated_0.k0bins,modulated_0.P_realizations[0])
axs[1].plot(modulated_1.k0bins,modulated_1.P_realizations[0])
axs[2].plot(modulated_2.k0bins,modulated_2.P_realizations[0])
plt.savefig("test_reconstr_class.png")
plt.show()
assert(1==0), "can I generate a single power spec with a non-identity primary beam?"

modulated_1.avg_realizations()
assert(1==0), "can I average over realizations? (requires generating boxes and power spectra and checking convergence)"


assert(1==0), "can I interpolate a converged power spec?" # could also test re-test interpolation independently by interpolating a P_fid or a single realization
assert(1==0), "can I generalize these tests to a cylindrically binned power spec?"

for i in range(Nrealiz):
    alert=Nrealiz//5
    if alert>0:
        if (i%alert==0):
            print("realization",i)
    ktest=np.linspace(twopi/Lsurvey,twopi*Nvox/Lsurvey,Nk)
    # if (i==0):


        # pb=PowerBox(Nvox,lambda k : k**idx, dim=3, boxlength=Lsurvey)
    _,T, _=generate_box(Ptest,ktest,Lsurvey,Nvox) # generate_box(P,k,Lsurvey,Nvox,primary_beam=False,primary_beam_args=False)
    _,T0,_=generate_box(Ptest,ktest,Lsurvey,Nvox, primary_beam=custom_response,primary_beam_args=bundled0)
    _,T1,_=generate_box(Ptest,ktest,Lsurvey,Nvox, primary_beam=custom_response,primary_beam_args=bundled1)
    _,T2,_=generate_box(Ptest,ktest,Lsurvey,Nvox, primary_beam=custom_response,primary_beam_args=bundled2)
    # Tpb=pb.delta_x()
    if i==0:
        V=Lsurvey**3
        fig,axs=plt.subplots(3,4,figsize=(20,15)) # version that adds ratios of the fiducial and reconstructed values
    Tmod0=T0*modulation0
    kfloors_mod,vals_mod0=generate_P(Tmod0,mode,Lsurvey,Nk, primary_beam=custom_response,primary_beam_args=bundled0) # generate_P(T, mode, Lsurvey, Nk0, Nk1=0, primary_beam=False,primary_beam_args=False) 
    allvals0[:,i]=vals_mod0
    Tmod1=T1*modulation1
    kfloors_mod,vals_mod1=generate_P(Tmod1,mode,Lsurvey,Nk, primary_beam=custom_response,primary_beam_args=bundled1)
    allvals1[:,i]=vals_mod1
    Tmod2=T2*modulation2
    kfloors_mod,vals_mod2=generate_P(Tmod2,mode,Lsurvey,Nk, primary_beam=custom_response,primary_beam_args=bundled2)
    allvals2[:,i]=vals_mod2
    kfloors,vals=generate_P(T,mode,Lsurvey,Nk)
    allvals[:,i]=vals
    for k in range(2):
        axs[k,0].scatter(kfloors,vals,color=colours[i])
        axs[k,1].scatter(kfloors_mod,vals_mod0,color=colours[i])
        axs[k,2].scatter(kfloors_mod,vals_mod1,color=colours[i])
        axs[k,3].scatter(kfloors_mod,vals_mod2,color=colours[i])
for i in range(3):
    for j in range(4):
        ylabels=["Power (K$^2$ Mpc$^3$)","Power (K$^2$ Mpc$^3$)","Ratio of powers (dimensionless, unitless)"]
        axs[i,j].set_xlabel("k (1/Mpc)")
        axs[i,j].set_ylabel(ylabels[i])
meanmean=np.mean(allvals,axis=-1)
mean0=np.mean(allvals0,axis=-1)
mean1=np.mean(allvals1,axis=-1)
mean2=np.mean(allvals2,axis=-1)
axs[0,0].plot(kfloors,meanmean, label="reconstructed")
axs[0,1].plot(kfloors,mean0,label="reconstructed")
axs[0,2].plot(kfloors,mean1,label="reconstructed")
axs[0,3].plot(kfloors,mean2,label="reconstructed")
axs[1,0].plot(kfloors,meanmean, label="reconstructed")
axs[1,1].plot(kfloors,mean0,label="reconstructed")
axs[1,2].plot(kfloors,mean1,label="reconstructed")
axs[1,3].plot(kfloors,mean2,label="reconstructed")

axs[2,0].plot(kfloors,kfloors**idx/meanmean)
axs[2,1].plot(kfloors,kfloors**idx/mean0)
axs[2,2].plot(kfloors,kfloors**idx/mean1)
axs[2,3].plot(kfloors,kfloors**idx/mean2)
for i in range(2):
    for j in range(4):
        axs[i,j].plot(kfloors,kfloors**idx,label="fiducial")
# if (power_spec_type=="pl"):
#     axs[2,0].plot(kfloors,kfloors**idx/meanmean)
#     axs[2,1].plot(kfloors,kfloors**idx/mean0)
#     axs[2,2].plot(kfloors,kfloors**idx/mean1)
#     axs[2,3].plot(kfloors,kfloors**idx/mean2)
#     for i in range(2):
#         for j in range(4):
#             axs[i,j].plot(kfloors,kfloors**idx,label="fiducial")
# elif (power_spec_type=="wn"):
#     for i in range(2):
#         for j in range(4):
#             axs[i,j].plot(kfloors,1+0*kfloors,label="fiducial") # like Ptest but "sampled" at kfloors
#     axs[2,0].plot(kfloors,1/meanmean)
#     axs[2,1].plot(kfloors,1/mean0)
#     axs[2,2].plot(kfloors,1/mean1)
#     axs[2,3].plot(kfloors,1/mean2)
axs[0,0].set_title("P(T) / power spec of unmodulated box")
axs[0,1].set_title("P(T*R) / power spec of response-modulated box \n(broad in config space)")
axs[0,2].set_title("P(T*R) / power spec of response-modulated box \n(medium in config space)")
axs[0,3].set_title("P(T*R) / power spec of response-modulated box \n(narrow in config space)")
for j in range(4):
    axs[1,j].set_title("inset for the case above")

for j in range(4):
    axs[2,j].set_title("Ratio of powers: fiducial/reconstructed")

plt.suptitle("Test spectral index={:4} P(k) calc for Lsurvey,Nvox,Nk,Nrealiz,sigma0**2,sigma1**2,sigma2**2={:4},{:4},{:4},{:4},{:4},{:4}".format(idx,Lsurvey,Nvox,Nk,Nrealiz,sigma02,sigma12,sigma22))
plt.legend()
plt.tight_layout()
plt.savefig("sph_"+str(int(10*idx))+"_"+mode+"_"+str(Nrealiz)+"realiz.png",dpi=750)
t1=time.time()
print("generating sph power spectra took",t1-t0,"s\n")
plt.show()