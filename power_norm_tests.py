from forecasting_pipeline import *

Lxy,Lz=115*u.Mpc,150*u.Mpc
Nvxy,Nvz=80,90
N_k_in=1500
k_in=np.linspace(1e-3,0.5,N_k_in)/u.Mpc
P_in=np.ones(N_k_in)*u.mK**2*u.Mpc**3

N_beam_pix=150
xy_ext,z_ext=60,80
beam_xy=np.linspace(-xy_ext,xy_ext,N_beam_pix)*u.Mpc
beam_z=np.linspace(-z_ext,z_ext,N_beam_pix)*u.Mpc
beam_x_grid,beam_y_grid,beam_z_grid=np.meshgrid(beam_xy.value,beam_xy.value,beam_z.value,indexing="ij")*u.Mpc
beam_evaled=np.exp(-(beam_x_grid.value**2+ 1.5*beam_y_grid.value**2+ 3*beam_z_grid.value**2)**2/10) # made up gaussian just for fun
beam_modes=(beam_xy.value,beam_xy.value,beam_z.value)

test_calc=cosmo_stats(Lxy,Lz,
                      P_fid=P_in,k_fid=k_in,
                      Nvox=Nvxy,Nvoxz=Nvz,
                      primary_beam_num=beam_evaled,primary_beam_type_num="manual",primary_beam_modes=beam_modes, 
                      radial_taper=kaiser,image_taper=kaiser,
                      frac_tol=0.5, no_monopole=True, 
                      fg_box=None)
test_calc.power_Monte_Carlo()
P=test_calc.P_binned_MC_complete
k_perp=test_calc.kperpbins
k_par=test_calc.kparbins

norm_to_use=CenteredNorm(vcenter=1., halfrange=0.2)

plt.figure()
plt.imshow(P.T,origin="lower",extent=[k_perp[0].value,k_perp[-1].value,k_par[0].value,k_par[-1].value],norm=norm_to_use)
plt.xlabel("kperp 1/Mpc")
plt.ylabel("kpar 1/Mpc")
plt.colorbar(extend="both")
plt.title("Power recovery test")
plt.savefig("power_recovery_test.png")
plt.close()