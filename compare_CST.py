from forecasting_pipeline import *

beam_sim_directory="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams_Aditya/CHORD_700/"


files=["feed_tilt_farfield_(f=0.7)_[1]_efield.txt",
       "fiducial_farfield_(f=0.7)_[1]_efield.txt"]
case_names=["feed tilt",
            "fiducial"]
power_cuts=[]
phis_of_interest=[-90.000,-45.000,0]

fig,axs=plt.subplots(3,3,figsize=(8,8),layout="constrained")
for i,file in enumerate(files):
    df = pd.read_table(beam_sim_directory+file, skiprows=[0, 1,], sep='\s+', 
                       names=['theta', 'phi', 'AbsE', 'AbsCr', 'PhCr', 'AbsCo', 'PhCo', 'AxRat'])
    nonlog=10**(df.AbsE.values/10) # non-log values
    theta_deg=df.theta.values
    theta=theta_deg*pi/180
    phi_deg=df.phi.values
    phi=phi_deg*pi/180

###
    for j,phi_of_interest in enumerate(phis_of_interest):
        print("i,j=",i,j)
        phi_equals_zero=np.nonzero(phi_deg==phi_of_interest)
        nonlog_cut=nonlog[phi_equals_zero]
        power_cut=nonlog_cut*np.conj(nonlog_cut)
        power_cut/=np.max(power_cut)
        power_cuts.append(power_cut)
        theta_cut=theta_deg[phi_equals_zero]
        axs[j,0].semilogy(theta_cut,power_cut,label=case_names[i])

        for k in range(3):
            axs[j,k].set_xlabel("boresight [rad]")
        axs[j,0].set_ylabel("power")
        axs[j,0].set_title("power beams phi="+str(phi_of_interest)+"deg")
        axs[j,0].legend()

        if i==1:
            axs[j,1].semilogy(theta_cut,power_cuts[1]/power_cuts[0],c="C2")
            axs[j,2].semilogy(theta_cut,power_cuts[1]-power_cuts[0],c="C3")
        axs[j,1].set_ylabel("dimensionless, unitless")
        axs[j,1].set_title("fiducial / feed tilt")

        axs[j,2].set_ylabel("power")
        axs[j,2].set_title("fiducial - feed tilt")

plt.savefig("CHORD_fidu_tilt_700_MHz.png")