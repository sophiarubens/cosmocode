import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import Blues
from astropy.io import fits

def primary_beam_crossing_time(nu,dec=30.,D=6.):
    beam_width_deg=1.029*(2.998e8/nu)/D*180/np.pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_mins= crossing_time_hrs_no_dec*np.cos(dec*np.pi/180)*60
    return crossing_time_mins

D=6.0
DRAO_lat=49.320791
elevations=[-30,0,30]
el_labels=["-30 deg el tilt","zenith at DRAO","+30 deg el tilt"]
FIRST="first_14dec17.fits"
hdulist=fits.open(FIRST)
# hdulist.info()
hdu1=hdulist[1]
# print("hdu1.header=\n",hdu1.header)
hdu1data=hdu1.data
ra_deg=   hdu1data["RA"]    # deg
dec_deg=  hdu1data["DEC"]   # deg
fpeak_mJy=hdu1data["FPEAK"] # mJy
hdulist.close()
N_northern_decs=256
northern_decs=np.linspace(0,90,N_northern_decs)
pb_crossing_times= [primary_beam_crossing_time(1500e6,dec=northern_dec) for northern_dec in northern_decs]
all_RAs_coarse=np.asarray([0,23])
pb_crossing_times_arr=np.vstack((pb_crossing_times,pb_crossing_times))
pcolor_ra,pcolor_dec=np.meshgrid(all_RAs_coarse,northern_decs,indexing="ij")

ra_hrs=ra_deg*24/360
lo=350
hi=1500
mid=(lo+hi)/2
sample_freqs=[lo,mid,hi] # MHz
c=2.998e8
threshold=5250
threshold_Jy=threshold/1000
filtered_ra_hrs=ra_hrs[fpeak_mJy>threshold]
filtered_ra_deg=ra_deg[fpeak_mJy>threshold]
filtered_dec=dec_deg[fpeak_mJy>threshold]
filtered_fluxes=fpeak_mJy[fpeak_mJy>threshold]
filtered_fluxes_Jy=filtered_fluxes/1000
N_sources=len(filtered_fluxes)
plt.figure(figsize=(20,8))
ax=plt.gca()
plt.pcolor(pcolor_ra,pcolor_dec,pb_crossing_times_arr,shading="nearest",label="conservative PB crossing times",cmap=Blues,alpha=0.2,edgecolors="face")
cbar=plt.colorbar()
cbar.ax.set_ylabel("crossing time for a 1500 MHz Gaussian beam (min)")
plt.scatter(filtered_ra_hrs,filtered_dec,c=filtered_fluxes_Jy,s=20,edgecolor="k",lw=0.5, vmin=np.percentile(filtered_fluxes_Jy,1),vmax=np.percentile(filtered_fluxes_Jy,85))
cbar=plt.colorbar()
cbar.ax.set_ylabel("peak point source flux (Jy)")
for i,freq in enumerate(sample_freqs):
    wl=c/(freq*1e6)
    beam_width=(1.22*wl/D)*180/np.pi
    plt.axhline(DRAO_lat+elevations[ 0]-beam_width,c="C"+str(2-i),label="FoV +beam width at "+str(freq)+" MHz")
    plt.axhline(DRAO_lat+elevations[-1]+beam_width,c="C"+str(2-i))
for i,el in enumerate(elevations):
    plt.axhline(DRAO_lat+el,c="C3") #,label=el_labels[i])
plt.text(0.5,20,"-30 deg el tilt",c="C3")
plt.text(0.5,50,"zenith at DRAO",c="C3")
plt.text(0.5,77,"+30 deg el tilt",c="C3")
plt.xlabel("RA (hr)")
plt.ylabel("dec (deg)")
secax=ax.secondary_xaxis("top",functions=(lambda val: val-8, lambda val: val+8))
secax.set_xlabel("Winter LST (UTC-8) (hrs)")
plt.ylim(0,90)
plt.xlim(0,24)
plt.title("1.4 GHz radio point sources brighter than {:4.3} Jy in the FIRST catalogue (N={:3})".format(threshold_Jy,N_sources))
plt.legend(loc=[0.65,0.25])
plt.savefig("FIRST_CHORD_brighter_than_"+str(threshold_Jy)+"_Jy.png")
plt.show()
###