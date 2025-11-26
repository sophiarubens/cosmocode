# global: 
realizations_scatter=False # ok enough for local tests, but would be eye-wateringly time-consuming and hard to inspect for Fir runs







# in calc_power_contamination
if realizations_scatter:
    Ptr_realizations=tr.P_realizations
    N_realiz=int(1/tr.frac_tol**2)
    orig_shape=(self.Nkpar_box,self.Nkperp_box)
    interpolated_tr=np.zeros((N_realiz,self.Nkpar_surv,self.Nkperp_surv))
    for i in range(N_realiz):
        interp_holder=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,P_fid=np.reshape(Ptr_realizations[i],orig_shape),Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                            Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                            k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                            no_monopole=self.no_monopole)
        interp_holder.interpolate_P(use_P_fid=True)
        interpolated_tr[i]=interp_holder.P_interp
    self.interpolated_tr=interpolated_tr






if realizations_scatter:
    Pth_realizations=th.P_realizations
    interpolated_th=np.zeros((N_realiz,self.Nkpar_surv,self.Nkperp_surv))
    for i in range(N_realiz):
        interp_holder=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,P_fid=np.reshape(Pth_realizations[i],orig_shape),Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                            Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                            k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                            no_monopole=self.no_monopole)
        interp_holder.interpolate_P(use_P_fid=True)
        interpolated_th[i]=interp_holder.P_interp
    self.interpolated_th=interpolated_th









# in cyl_sph_plots
    if realizations_scatter:
        interpolated_th=windowed_survey.interpolated_th
        interpolated_tr=windowed_survey.interpolated_tr
        flat_shape=windowed_survey.Nkpar_surv*windowed_survey.Nkperp_surv
        for i in range(windowed_survey.maxiter):
            axs[0].scatter(np.reshape(kcyl_mags_for_interp_flat,flat_shape),np.reshape(interpolated_tr[i],flat_shape),s=0.1,c="C1")
            axs[0].scatter(np.reshape(kcyl_mags_for_interp_flat,flat_shape),np.reshape(interpolated_th[i],flat_shape),s=0.1,c="C0")