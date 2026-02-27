import numpy as np

a=np.array([1,5,8,2,7,6])
N=3
seed=49

rng=np.random.default_rng(seed)

for i in range(N):
    legacy=np.random.normal(loc=0.*a,scale=a)
    modern=rng.normal(loc=0.*a,scale=a)
    print("legacy   seeded RNs = ",legacy)
    print("modern unseeded RNs = ",modern,"\n")
print("\n\n")