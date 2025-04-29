import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j0,j1 # 0th- and 1st-order Bessel functions of the first kind
from scipy.integrate import quad,dblquad
import time

pi=np.pi
twopi=2.*pi
ln2=np.log(2)
inf=np.infty

def thetaHWHM_to_alpha(thetaHWHM):
    npts=2222
    perp_vals=np.linspace(0,twopi,npts)
    basic_airy_beam=(j1(perp_vals)/perp_vals)**2
    basic_airy_beam_half_max=1./8. # derived
    thetaHWHM_ref_airy=perp_vals[np.nanargmin(np.abs(basic_airy_beam-basic_airy_beam_half_max))]
    return thetaHWHM_ref_airy/thetaHWHM

def airy_arg(rperp,kperp,alpha):
    airyarg=alpha*rperp
    return (j1(airyarg)/airyarg)**2*j0(rperp*kperp)*rperp

def airy_entry(kperp,alpha):
    integral,_=quad(airy_arg,0,np.infty,args=(kperp,alpha,)) # args are the variables in the integrand apart from the variable of integration (first n things passed to the innermost layer of the wrap structure)
    return integral**2

def airy(kperp_vec,alpha):
    nk=len(kperp_vec)
    Wperp_vec=np.zeros(nk)
    for i,kperp_val in enumerate(kperp_vec):
        Wperp_vec[i]=airy_entry(kperp_val,alpha)
    return Wperp_vec

def calc_par_vec(deltakparvec,sigLoS,r0):
    prefac=(sigLoS**4-2*deltakparvec**2*sigLoS**6+2*r0*sigLoS**2+deltakparvec**4*sigLoS**8+r0**4+2*deltakparvec**2*sigLoS**4*r0**2)
    expterm=twopi*sigLoS**2*np.exp(-deltakparvec**2*sigLoS**2)
    return prefac*expterm

def calc_perp_vec(deltakperpvec,Dc,thetaHWHM,beamtype="Gaussian"):
    beamtype=beamtype.lower()
    if beamtype=="gaussian":
        alpha=ln2/(thetaHWHM*Dc)**2
        print("Gaussian alpha=ln2/(thetaHWHM*Dc)**2=",alpha)
        vec=np.exp(-deltakperpvec**2/(2*alpha))
    elif beamtype=="airy":
        nk=len(deltakperpvec)
        vec=np.zeros(nk)
        alpha=thetaHWHM_to_alpha(thetaHWHM)
        print("Airy alpha=thetaHWHM_to_alpha(thetaHWHM)=",alpha)
        vec=airy(deltakperpvec,alpha)
    else: 
        assert(1==0), "currently supported beam types are Airy and Gaussian"
    return vec

def W_cyl_binned(deltakparvec,deltakperpvec,sigLoS,r0,thetaHWHM,save=False,savename="test",btype="Gaussian"):
    par_vec=calc_par_vec(deltakparvec,sigLoS,r0)
    # perp_vec=calc_perp_vec(deltakperpvec,beamtype,thetaHWHM)
    perp_vec=calc_perp_vec(deltakperpvec,r0,thetaHWHM,beamtype=btype)
    fig,axs=plt.subplots(2,3,figsize=(20,10))
    axs[0,0].plot(deltakparvec,par_vec)
    axs[0,0].set_xlabel("k$_{||}$")
    axs[0,0].set_ylabel("not-yet-normalized amplitude")
    axs[0,0].set_title("parallel term")
    axs[0,1].plot(deltakperpvec,perp_vec)
    axs[0,1].set_xlabel("k$_\perp$")
    axs[0,1].set_ylabel("not-yet-normalized amplitude")
    axs[0,1].set_title("perp term")
    par_arr,perp_arr=np.meshgrid(par_vec,perp_vec)
    im=axs[1,0].imshow(par_arr,extent=[deltakparvec[0],deltakparvec[-1],deltakparvec[-1],deltakparvec[0]]) # ,norm="log")
    fig.colorbar(im,ax=axs[1,0])
    axs[1,0].set_xlabel("k$_{||}$")
    axs[1,0].set_ylabel("arbitrary")
    axs[1,0].set_title("parallel meshgridded array")
    im=axs[1,1].imshow(perp_arr,extent=[deltakperpvec[0],deltakperpvec[-1],deltakperpvec[-1],deltakperpvec[0]]) # ,norm="log")
    fig.colorbar(im,ax=axs[1,1])
    axs[1,1].set_xlabel("arbitrary")
    axs[1,1].set_ylabel("k$_{\perp}$")
    axs[1,1].set_title("perp meshgridded array")
    meshed=par_arr*perp_arr # interested in elementwise (not matrix) multiplication
    normed=meshed/np.sum(meshed)
    im=axs[1,2].imshow(normed,extent=[deltakperpvec[0],deltakperpvec[-1],deltakparvec[-1],deltakparvec[0]],aspect=deltakperpvec[-1]/deltakparvec[-1]) # ,norm="log")
    axs[1,2].set_xlabel("k$_{||}$")
    axs[1,2].set_ylabel("k$_{\perp}$")
    fig.colorbar(im,ax=axs[1,2])
    axs[1,2].set_title("normalized product of parallel and perp arrays")
    for i in range(2):
        for j in range(3):
            plt.setp(axs[i,j].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.suptitle(savename+" cylindrically binned window check")
    plt.tight_layout()
    plt.savefig(savename+"_in_wrapper_cyl_binned_window_check.png")
    plt.show()
    if (save):
        np.save('W_cyl_binned_2D_proxy'+str(time.time())+'.txt',normed)
    return normed