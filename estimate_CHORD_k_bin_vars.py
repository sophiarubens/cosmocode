import numpy as np
from matplotlib import pyplot as plt
from astropy import units as un
import py21cmsense
from py21cmsense import GaussianBeam, Observation, Observatory, PowerSpectrum



def diag_stripe_arr(row):
    dim=len(row)
    arr=np.zeros((dim,dim))
    for k in range(dim): # k=i+j
        for j in range(dim):
            i=k-j
            arr[i,j]=row[k]
    mod=np.tril(-np.ones((dim,dim)))
    mod+=np.eye(dim)
    mod=np.flip(mod,axis=1)
    return arr+mod

print("observatories built into 21cmSense:",py21cmsense.observatory.get_builtin_profiles())
pstart="/Users/sophiarubens/Downloads/research"
path_to_chord_ant_pos_211208="/CHORD_all_telecon_materials/chord_dish_coords_fences_wgap_clean_trimmed_8Dec2021.txt"
path_to_working_dir="/code/param_bias"
chord_x,chord_y=np.genfromtxt(pstart+path_to_chord_ant_pos_211208).T
nants=len(chord_x)
print("nants check:",nants)
CHORD_appx_unif_elev=546.5
chord_z_appx=CHORD_appx_unif_elev*np.ones(nants)

chord_appx_antpos=np.vstack([chord_x,chord_y,chord_z_appx]).T*un.m
print("chord_appx_antpos.shape=",chord_appx_antpos.shape)

plt.figure()
p=plt.scatter(chord_x,chord_y,s=0.5)
plt.xlabel("northing (m)")
plt.ylabel("easting (m)")
plt.title("CHORD antenna positions as of 2021-12-08")
plt.show()

k_21_900MHz=np.load(pstart+path_to_working_dir+"/camb_k.npy")
P_21_900MHz=np.load(pstart+path_to_working_dir+"/camb_P.npy")

beam_test=GaussianBeam(frequency=900. * un.MHz, dish_size=6 * un.m)
print("beam.frequency=",beam_test.frequency)

chord_sensitivity = PowerSpectrum(
    observation=Observation( # higher-order correction: 21cmSense defaults to Planck15, but elsewhere I'm using Planck18 (would be good to eventually remove the discrepancy)
        observatory=Observatory(
            antpos=chord_appx_antpos,
            beam=beam_test,
            latitude=49.321 * un.deg,
        )
    )
    # ),
    # k_21=k_21_900MHz,
    # delta_21=k_21_900MHz**3*P_21_900MHz/(2*np.pi**2),
)

# print("chord_sensitivity.k_21=",chord_sensitivity.k_21)
# chord_sensitivity..k_21=k_21_900MHz
# chord_sensitivity.del
# assert(1==0),"debug— figuring out where the pspec is stored"

chord_21cmse_k1d=chord_sensitivity.k1d
chord_21cmse_sigk = chord_sensitivity.calculate_sensitivity_1d() 
plt.figure()
plt.plot(chord_21cmse_k1d, chord_21cmse_sigk)
plt.xlabel("k [h/Mpc]")
plt.ylabel(r"$\delta \Delta^2_{21}$")
plt.yscale("log")
plt.xscale("log")
plt.title("**NOT YET THE** CHORD 1D sensitivity")
plt.show()

np.save("chord_21cmse_k1d.npy",np.array(chord_21cmse_k1d))
np.save("chord_21cmse_sigk.npy",np.array(chord_21cmse_sigk))