from forecasting_pipeline import *

CST_dir=CST_dir="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams/CHORD_feed_tilts_integ_dom_600/farfield_"
p1="pol1/f_"
p2="pol2/f_"
lo=0.5998*u.GHz
hi=0.6004*u.GHz
mid=(lo+hi)/2
Delta=200*u.kHz
CST_freqs=np.arange(lo.value,hi.value,Delta.value)*Delta.unit
k_perp=kperp(mid,6.3*u.m,np.sqrt((6.3*7)**2+(8.5*10)**2)*u.m)
kperpmin=k_perp[0]
Lbox_xy=twopi/kperpmin
Nbox_xy=256 # in the pipeline, this needs to be downstream-compatible with the intended cosmo_stats calls, but for this standalone test it's more or less arbitrary
precalculated_xy_vec=Lbox_xy*fftshift(fftfreq(Nbox_xy))
mini_beam=reconfigure_CST_beam(lo,hi,Delta,Nxy=256,
                               beam_sim_directory=CST_dir,f_head="fiducial/",
                               f_mid1=p1,f_mid2=p2,f_tail="_GHz.txt",box_outname="mini_box_fiducial")

mini_beam.gen_box_from_simulated_beams()
mini_box=mini_beam.box
oneslice=mini_box[:,:,0]
plt.figure()
plt.imshow(oneslice.T,
           origin="lower",norm="log")
plt.colorbar()
plt.savefig("re_slice_cst.png")
plt.close()

mini_fidu_per_antenna_ified=synthesize_beam(mode="pathfinder",N_timesteps=1,
                                    N_pbws_pert=0,nu_ctr=mid,N_grid_pix=256,
                                    distribution="random",
                                    sub_ensemble_of_CST_beams=mini_box,
                                    CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs)
mini_fidu_per_antenna_ified.stack_to_box()
pa_ified_mini_box=mini_fidu_per_antenna_ified.box
pa_ified_oneslice=pa_ified_mini_box[:,:,0]
fig,axs=plt.subplots(1,2,layout="constrained")
ln=LogNorm(vmin=1e-10,vmax=1)
im=axs[0].imshow(pa_ified_oneslice.T,
           origin="lower",norm=ln)
plt.colorbar(im,ax=axs[0])
im=axs[1].imshow(-pa_ified_oneslice.T,
           origin="lower",norm=ln)
plt.colorbar(im,ax=axs[1])
plt.savefig("mini_box_per_antenna_ified_box.png",dpi=500)
plt.close()