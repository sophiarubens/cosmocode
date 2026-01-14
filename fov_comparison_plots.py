import numpy as np
from matplotlib import pyplot as plt
from cosmo_distances import *

Nb512= 1010 # CHORD-512 (as long as receiver hut gaps remove redundancy only and not unique baselines, as I'm sure they planned)
Nb64=  123  # CHORD-64  (same caveat about the gaps)
b_NS=8.5 # m
N_NS=24
b_EW=6.3 # m
N_EW=22
bmin=b_EW
bmax512=np.sqrt((N_NS*b_NS)**2+(N_EW*b_EW)**2)
bmax64=bmax512/2.

freqs=np.arange(300,1420,200)
Ncases=len(freqs)
colours=plt.cm.Greens(np.linspace(1,0.2,Ncases))
chanw=0.183

def get_channel_config(nu_ctr,Deltanu,evol_restriction_threshold=1./15.):
    """
    args
    nu_ctr                     = central frequency of the survey
    Deltanu                    = channel width
    evol_restriction_threshold = $N\Delta\nu/\nu ~ \Delta z/z ~$ evol_restriction_threshold (1/15 common in some HERA surveys); N = number of channels in the survey

    returns
    NDeltanu = survey bandwidth
    N        = number survey channels
    """
    NDeltanu=nu_ctr*evol_restriction_threshold
    N=NDeltanu/Deltanu
    return NDeltanu,N

case_names=[]
case_k=[]
for i,freqi in enumerate(freqs):
    _,Ni=get_channel_config(freqi,chanw)
    kpari=kpar(freqi,chanw,Ni)
    kperp64i=kperp(freqi,Nb64,bmin,bmax64)
    kperp512i=kperp(freqi,Nb512,bmin,bmax512)
    case_names.append(str(int(freqi))+" MHz (z="+str(round(freq2z(nu21,freqi),3))+")")
    case_k.append([kpari,kperp512i])
    case_k.append([kpari,kperp64i])

conservative_breakdown=0.2
optimistic_breakdown=  1.
npts=200

def quarter_torus_piecewise_curves(lo,md,hi,npts=200):
    x1=np.linspace(lo,md,npts)
    y_inner_1=np.sqrt(md**2-x1**2)
    y_outer_1=np.sqrt(hi**2-x1**2)
    x2=np.linspace(md,hi,npts)
    y_outer_2=np.sqrt(hi**2-x2**2)
    return x1,y_inner_1,y_outer_1,x2,y_outer_2

x_fill_bw,y_arc_cons,y_arc_opti,x_fill_bw_2,y_arc_opti_2=quarter_torus_piecewise_curves(0.,conservative_breakdown,optimistic_breakdown)

BAOlo,BAOhi=np.pi*np.array([0.67,0.74])/55.
xfill1,yin1,yout1,xfill2,yout2=quarter_torus_piecewise_curves(0.,BAOlo,BAOhi)
eqlo,eqhi=0.02*np.array([0.67,0.74])
xfill1eq,yin1eq,yout1eq,xfill2eq,yout2eq=quarter_torus_piecewise_curves(0.,eqlo,eqhi)

torusalpha=0.5

Nsubplots=2
fig,axs=plt.subplots(1,Nsubplots,layout="constrained")
kperp_for_wedge_calc=np.linspace(0,120,3000)
for j in range(Nsubplots):
    if (j>0):
        hide_freq_label="_"
        hide_th_l=""
    else:
        hide_freq_label=""
        hide_th_l="_"
    for i,case in enumerate(case_k):
        kpar,kperp=case
        zeroskperp=np.zeros(len(kperp))
        kparstart=kpar[0]
        kparstop=kpar[-1]
        colour=colours[i//2]
        if (i%2==0):
            axs[j].fill_between(kperp,zeroskperp+kparstart,y2=zeroskperp+kparstop,facecolor="none",lw=1,ec=colour,label=hide_freq_label+case_names[i//2])
            wedge=wedge_kpar(freqs[i//2],kperp_for_wedge_calc) # wedge_kpar(nu_ctr,kperp,H0=H0_Planck18,nu_rest=nu21)
            axs[j].plot(kperp_for_wedge_calc,wedge,ls=(0,(5,10)),lw=0.5,c=colour) # this line style is what the pyplot docs recommend for a "loosely dashed" line
        else:
            axs[j].fill_between(kperp,zeroskperp+kparstart,y2=zeroskperp+kparstop,facecolor=colour,lw=1,ec=colour,alpha=0.1) # ,facecolor="none",lw=1,ec=colour,hatch="x")
            axs[j].fill_between(kperp,zeroskperp+kparstart,y2=zeroskperp+kparstop,facecolor="none",lw=1,ec=colour          ) # overplot the boundary with no fill so it is not affected by transparency
    axs[j].fill_between(x_fill_bw,y_arc_cons,y2=y_arc_opti,label=hide_th_l+"~ linear theory breakdown",color="c",          alpha=torusalpha,edgecolor=None,linewidth=0)
    axs[j].fill_between(x_fill_bw_2,y_arc_opti_2,                                                      color="c",          alpha=torusalpha,edgecolor=None,linewidth=0) # second fill between bounded by 0 and the the optimistic curve
    axs[j].fill_between(xfill1,yin1,y2=yout1,              label=hide_th_l+"~ first BAO wiggle",                    color="tab:purple", alpha=torusalpha,edgecolor=None,linewidth=0)
    axs[j].fill_between(xfill2,yout2,                                                                  color="tab:purple", alpha=torusalpha,edgecolor=None,linewidth=0)
    axs[j].fill_between(xfill1eq,yin1eq,y2=yout1eq,        label=hide_th_l+"~ k$_\mathrm{eq}$",        color="r",          alpha=torusalpha,edgecolor=None,linewidth=0)
    axs[j].fill_between(xfill2eq,yout2eq,                                                              color="r",          alpha=torusalpha,edgecolor=None,linewidth=0)

    axs[j].set_xlabel("k$_\perp$ (Mpc$^{-1}$)")
    axs[j].set_ylabel("k$_{||}$ (Mpc$^{-1}$)")
    axs[j].set_aspect(1)
    axs[j].set_box_aspect(1)
axs[0].legend()
axs[0].text(1,25,"- - = upper extent of \n        foreground wedge")
axs[0].set_xbound(-1,120)
axs[0].set_ybound(-1,120)
axs[0].set_title("complete view")
axs[1].set_xbound(-0.01,1.01*conservative_breakdown)
axs[1].set_ybound(-0.01,1.01*conservative_breakdown)
axs[1].set_title("inset")
axs[1].legend()
plt.suptitle("Comparing the CHORD-accessible parts of Fourier space to theory\n" \
             "predictions for selected frequencies across the CHORD band")
plt.savefig("fov_theory_comp.png",dpi=600)