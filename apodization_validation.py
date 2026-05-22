from scipy.signal.windows  import kaiser
from scipy.fft import fftfreq

print("kaiser(5,14)=",kaiser(5,14))

print("kaiser(fftfreq(5),14)=",kaiser(fftfreq(5),14))