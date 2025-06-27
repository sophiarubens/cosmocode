import numpy as np
from matplotlib import pyplot as plt
from cosmo_distances import *
from bias_helper_fcns import *

# kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18)
# kperp(nu_ctr,N_baselines,bmin,bmax)
# get_channel_config(nu_ctr,Deltanu,evol_restriction_threshold=1./15.) returns NDeltanu,N

Nb512= 1010 # CHORD-512 (as long as receiver hut gaps remove redundancy only and not unique baselines, as I'm sure they planned)
Nb64=  123  # CHORD-64  (same caveat about the gaps)
b_NS=8.5 # m
N_NS=24
b_EW=6.3 # m
N_EW=22
bmin=b_EW
bmax512=np.sqrt((N_NS*b_NS)**2+(N_EW*b_EW)**2)
bmax64=bmax512/2.

freqs=np.arange(300,1420,110)
Ncases=len(freqs)
colours=plt.cm.Greens(np.linspace(1,0.2,Ncases))
chanw=0.183

case_names=[]
case_k=[]
for i,freqi in enumerate(freqs):
    _,Ni=get_channel_config(freqi,chanw)
    kpari=kpar(freqi,chanw,Ni)
    kperp64i=kperp(freqi,Nb64,bmin,bmax64)
    kperp512i=kperp(freqi,Nb512,bmin,bmax512)
    case_names.append(str(int(freqi))+" MHz")
    case_names.append(str(int(freqi))+" MHz")
    case_k.append([kpari,kperp512i])
    case_k.append([kpari,kperp64i])

conservative_breakdown=0.2
optimistic_breakdown=  1.
npts=200

x_fill_bw=np.linspace(0,conservative_breakdown,  npts)
y_arc_cons=np.sqrt(conservative_breakdown**2-x_fill_bw**2)
y_arc_opti=np.sqrt(optimistic_breakdown**2-  x_fill_bw**2)

x_fill_bw_2=np.linspace(conservative_breakdown,optimistic_breakdown,npts)
y_arc_opti_2=np.sqrt(optimistic_breakdown**2-x_fill_bw_2**2)
torusalpha=0.5

Nsubplots=4
fig,axs=plt.subplots(1,Nsubplots,figsize=(20,5))
for j in range(Nsubplots):
    for i,case in enumerate(case_k):
        kpar,kperp=case
        zeroskperp=np.zeros(len(kperp))
        kparstart=kpar[0]
        kparstop=kpar[-1]
        colour=colours[i//2]
        if (i%2==0):
            axs[j].fill_between(kperp,zeroskperp+kparstart,y2=zeroskperp+kparstop,facecolor="none",lw=1,ec=colour)
        else:
            axs[j].fill_between(kperp,zeroskperp+kparstart,y2=zeroskperp+kparstop,                    facecolor=colour,lw=1,ec=colour,alpha=0.1) # ,facecolor="none",lw=1,ec=colour,hatch="x")
            axs[j].fill_between(kperp,zeroskperp+kparstart,y2=zeroskperp+kparstop,label=case_names[i],facecolor="none",lw=1,ec=colour          ) # overplot the boundary with no fill so it is not affected by transparency
    axs[j].fill_between(x_fill_bw,y_arc_cons,y2=y_arc_opti,label="linear theory breaks down\nsomewhere in this range",color="c",alpha=torusalpha,edgecolor=None,linewidth=0)
    axs[j].fill_between(x_fill_bw_2,y_arc_opti_2,                                                                     color="c",alpha=torusalpha,edgecolor=None,linewidth=0) # second fill between bounded by 0 and the the optimistic curve

    axs[j].set_xlabel("k$_\perp$")
    axs[j].set_ylabel("k$_{||}$")
axs[0].legend()
axs[0].set_title("full plot")
axs[1].set_title("slight inset")
axs[1].set_xbound(-0.1,4)
axs[1].set_ybound(-0.1,4)
axs[2].set_title("moderate inset")
axs[2].set_xbound(-0.01,1.01*optimistic_breakdown)
axs[2].set_ybound(-0.01,1.01*optimistic_breakdown)
axs[3].set_title("drastic inset")
axs[3].set_xbound(-0.01,1.01*conservative_breakdown)
axs[3].set_ybound(-0.01,1.01*conservative_breakdown)
# plt.axis("equal")
plt.suptitle("CHORD field of view comparison")
plt.tight_layout()
plt.savefig("fov_theory_comp.png")
plt.show()