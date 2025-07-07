import numpy as np

pi=np.pi
twopi=2.*pi
Lsurvey=126
Nvox=52
Delta=Lsurvey/Nvox

k_vec_for_box= twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta)) # as long as the origin is where I want it to be (i.e. the state of fftshiftedness is correct), this seems justified
ifft_of_k_vec_for_box=np.fft.irfft(k_vec_for_box)
twopi_div_k_vec_for_box=twopi/k_vec_for_box
rmags=Lsurvey*np.fft.fftfreq(Nvox) # line that currently lives in generate_box but that I suspect is wrong
pi_div_minus=pi/(twopi/Lsurvey-k_vec_for_box)

print("k_vec_for_box=          ",k_vec_for_box)
# print("ifft_of_k_vec_for_box=  ",ifft_of_k_vec_for_box)
print("twopi_div_k_vec_for_box=",twopi_div_k_vec_for_box)
print("rmags=                  ",rmags)
print("twopi_div_minus=        ",pi_div_minus)