import numpy as np

sky_plane_sigmas,sigLoS,Deltabox= np.array([2.73045741, 2.73045741]), 4.1489169284214595, 2.4212660143273936

print("sky_plane_sigmas<Deltabox:",sky_plane_sigmas<Deltabox) # can't do this bc ambiguous... must use np.any or np.all
print("np.any(sky_plane_sigmas)<Deltabox",np.any(sky_plane_sigmas)<Deltabox) # why does this return True when, elementwise, it is objectively false!!?
all_sigmas=np.concatenate((sky_plane_sigmas,[sigLoS]))
print("np.any(all_sigmas<Deltabox):",np.any(all_sigmas<Deltabox))
# sky_plane_sigmas[0]<Deltabox: False
# sky_plane_sigmas[1]<Deltabox: False
# (np.any(sky_plane_sigmas)<Deltabox): True
# (sigLoS<Deltabox): False
# (np.any(sky_plane_sigmas)<Deltabox) or (sigLoS<Deltabox): True