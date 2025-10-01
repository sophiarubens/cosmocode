import numpy as np
from matplotlib import pyplot as plt
from CHORD_vis import *

def getrms(x):
    return np.sqrt(np.mean(x**2))

test_freq=885
lambda_obs=c/(test_freq*1e6)
z_obs=nu_HI_z0/test_freq-1.

N_pbs_to_pert=100
max_N_perts=16
N_pert_types_cases=np.arange(2,max_N_perts,2,dtype="int")
N_cases_N_pert_types=N_pert_types_cases.shape[0]
all_perts_horiz=np.arange(0,max_N_perts,2)

N_iterations=25
fidu=CHORD_image(N_pert_types=0)
pb02=CHORD_image(N_pert_types=2,  num_pbws_to_pert=N_pbs_to_pert)
pb04=CHORD_image(N_pert_types=4,  num_pbws_to_pert=N_pbs_to_pert)
pb06=CHORD_image(N_pert_types=6,  num_pbws_to_pert=N_pbs_to_pert)
pb08=CHORD_image(N_pert_types=8,  num_pbws_to_pert=N_pbs_to_pert)
pb10=CHORD_image(N_pert_types=10, num_pbws_to_pert=N_pbs_to_pert)
pb12=CHORD_image(N_pert_types=12, num_pbws_to_pert=N_pbs_to_pert)
pb14=CHORD_image(N_pert_types=14, num_pbws_to_pert=N_pbs_to_pert)
cases=[fidu,pb02,pb04,pb06,pb08,pb10,pb12,pb14]
time_trials=np.zeros((N_iterations,8))
stats_per_case=np.zeros((N_iterations,N_cases_N_pert_types+1,3)) # rows for perturbation cases; columns for rms of img, ratio, and residual
for i in range(N_iterations):
    for j,case in enumerate(cases):
        t0=time.time()
        dimg=case.calc_dirty_image()
        t1=time.time()

        time_trials[i,j]=t1-t0
        if j==0:
            ratiorms=np.nan
            residualrms=np.nan
            # fiduimg=case.dirty_image
            fiduimg=dimg
            caserms=getrms(fiduimg)
        else:
            # caseimg=case.dirty_image
            caseimg=dimg
            caserms=getrms(caseimg)
            ratio=caseimg/fiduimg
            ratiorms=getrms(ratio[~np.isnan(ratio)])
            residual=caseimg-fiduimg
            residualrms=getrms(residual)

        stats_per_case[i,j,:]=[caserms,ratiorms,residualrms]
np.save("time_trials"+str(time.time())+".npy",time_trials)

with open("stats_per_case.txt", "w") as f:
    for i, slice2d in enumerate(stats_per_case):
        np.savetxt(f, slice2d)
        if i < N_iterations - 1:
            f.write("\n")

avg_times=np.mean(time_trials,axis=0)
pertvals=np.linspace(0,max_N_perts-2,100)
quadfit=np.poly1d(np.polyfit(all_perts_horiz,avg_times,2))

fig,axs=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[4,1]},sharex=True)
axs[0].plot(all_perts_horiz,avg_times,c="C1",label="average of trials")
axs[0].plot(pertvals,quadfit(pertvals),c="C0",label="quadratic fit")
for i in range(N_iterations):
    if i==(N_iterations-1):
        labelhere="per-trial times"
    else:
        labelhere=""
    axs[0].scatter(all_perts_horiz,time_trials[i],c="C2",s=0.5,label=labelhere)
axs[1].set_xlabel("number of primary beam width perturbation types")
axs[0].set_ylabel("time to evaluate (s)")
axs[1].set_ylabel("residual eval time (s)")
axs[1].scatter(all_perts_horiz,avg_times-quadfit(all_perts_horiz),s=0.5)
plt.suptitle("Time scaling of dirty image calc with \nnumber of primary beam perturbation types")
axs[0].legend()
plt.tight_layout()
plt.savefig("time_scaling_dirty_image_calc.png")
plt.show()

fig,axs=plt.subplots(1,3,figsize=(12,5))
all_perts_horiz=np.arange(0,max_N_perts,2)
for i in range(N_iterations):
    axs[0].scatter(all_perts_horiz,stats_per_case[i,:,0],c="C0",s=0.5) # (N_iterations,N_cases_N_pert_types+1,3)
    axs[1].scatter(all_perts_horiz,stats_per_case[i,:,1],c="C0",s=0.5)
    axs[2].scatter(all_perts_horiz,stats_per_case[i,:,2],c="C0",s=0.5)
prefixes=["image","ratio","residual"]
for i in range(3):
    axs[i].set_title(prefixes[i]+" rms")
    axs[i].set_xlabel("# primary beam perturbation types")
    axs[i].set_ylabel("dimensionless")
plt.suptitle("Figure of merit for how imaging errors scale with the number of kinds of PBW perturbation")
plt.tight_layout()
plt.savefig("more_trials_how_errs_scale_w_num_pbw_pert_types.png")
plt.show()