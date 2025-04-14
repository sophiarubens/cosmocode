import numpy as np
from matplotlib import pyplot as plt
from astropy import units as un
import py21cmsense
from py21cmsense import GaussianBeam, Observation, Observatory, PowerSpectrum, hera

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

hera_sensitivity = PowerSpectrum(
    observation=Observation(
        observatory=Observatory(
            antpos=hera(hex_num=7, separation=14 * un.m),
            beam=GaussianBeam(frequency=135.0 * un.MHz, dish_size=14 * un.m),
            latitude=38 * un.deg,
        )
    )
)

k_21_900MHz=np.load(pstart+path_to_working_dir+"/camb_k.npy")
P_21_900MHz=np.load(pstart+path_to_working_dir+"/camb_P.npy")

def delta_from_P(k,P):
    return k**3*P/(2.*np.pi**2)
delta2_21_900MHz=delta_from_P(k_21_900MHz,P_21_900MHz)

chord_sensitivity = PowerSpectrum(
    observation=Observation(
        observatory=Observatory(
            antpos=chord_appx_antpos,
            beam=GaussianBeam(frequency=900. * un.MHz, dish_size=6 * un.m),
            latitude=49.321 * un.deg,
        )
    )
    # ),
    # k_21=k_21_900MHz,
    # delta_21=delta2_21_900MHz # kwarg name is ambiguous but the docs say this is, indeed, Delta^2 
)

# ## # the py21cmSense tutorial "changes the redshift" of a sensitivity calculation by changing the frequency in the beam object used for setup
# redshifts = np.linspace(5, 25, 25)
# for i, z in enumerate(redshifts):
#     sense_high = sensitivity.clone(
#         observation=observation.clone(
#             observatory=observation.observatory.clone(
#                 beam=p21s.GaussianBeam(frequency=1420 * units.MHz / (1 + z), dish_size=14 * units.m)
#             )
#         )
#     )

#     ax.flatten()[i].plot(
#         sense_high.k1d, sense_high.calculate_sensitivity_1d(), label="sample+thermal"
#     )
# ##

power_std = chord_sensitivity.calculate_sensitivity_1d()
plt.figure()
plt.plot(chord_sensitivity.k1d, power_std)
plt.xlabel("k [h/Mpc]")
plt.ylabel(r"$\delta \Delta^2_{21}$")
plt.yscale("log")
plt.xscale("log")
plt.title("**NOT YET THE** CHORD 1D sensitivity")
plt.show()

# 49.320902, -119.621864 # middle-ish of the CHORD site, according to the pin I dropped using the satellite view of Google Maps
# 49.321,    -119.622    # still in the middle-ish of the CHORD site
# ns_step=8.5 # all spatial values here are in m
# ew_step=6.3
# northing_start=5.466e6
# easting_start=309420
# n_dishes_ns=24
# n_dishes_ew=22
# CHORD_appx_unif_elev=546.5
# CHORD_ish_x=np.arange(easting_start, easting_start+ n_dishes_ew*ew_step,ew_step)
# CHORD_ish_y=np.arange(northing_start,northing_start+n_dishes_ns*ns_step,ns_step)
# CHORD_ish_X,CHORD_ish_Y=np.meshgrid(CHORD_ish_x,CHORD_ish_y)
# print("CHORD_ish_X.shape=",CHORD_ish_X.shape)
# print("CHORD_ish_Y.shape=",CHORD_ish_Y.shape)
# CHORD_ish_Z=CHORD_appx_unif_elev*np.ones((n_dishes_ns,n_dishes_ew))
# print("CHORD_ish_Z.shape=",CHORD_ish_Z.shape)
# CHORD_ish_dishes=np.dstack([CHORD_ish_X,CHORD_ish_Y,CHORD_ish_Z])

# plt.figure()
# p=plt.scatter(CHORD_ish_X,CHORD_ish_Y,CHORD_ish_Z)
# p.set_sizes([0.5])
# plt.xlabel("northing (m)")
# plt.ylabel("easting (m)")
# plt.title("Oversimplified CHORD")
# plt.colorbar()
# plt.show()