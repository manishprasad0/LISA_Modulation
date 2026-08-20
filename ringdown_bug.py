import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from lisatools.utils.constants import *
from bbhx.waveformbuild import BBHWaveformFD


DAY_SI = 24*3600
MONTH_SI = YRSID_SI/12


wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))

m1 = 1e5
m2 = 2e5
chi1z = 0.0
chi2z = 0.7
dist = 2 * 1e9 * PC_SI
phi_ref = 1.0
f_ref = 0.0
inc = 2*np.pi/3  #2*np.pi/3.
lam = 0
beta= 0
psi = 3*np.pi/8
t_ref = 6*MONTH_SI

modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
Tobs = YRSID_SI

dt = 5.
N = int(Tobs / dt)
Tobs = N * dt
freq = np.fft.rfftfreq(N,dt)
freq[0] = freq[1] # have to do this as freq[0] = 0 causes issues with the waveform generator

wave_freq_domain = wave_gen(m1, m2, chi1z, chi2z,
                            dist, phi_ref, f_ref, inc, lam,
                            beta, psi, t_ref, 
                            freqs=freq, modes=modes, 
                            direct=False, fill=True, squeeze=True, length=1024)[0]

wave_time_domain = sp.fft.irfft(wave_freq_domain, axis=-1)
time_array = (np.arange(len(wave_time_domain[1]))*dt)/MONTH_SI

idx_start = len(time_array) // 2  + 100000

plt.plot(time_array[idx_start:], (wave_time_domain[1][idx_start:]))


# The ampliduge after ringdown starts increasing at the end of the observation time, which is unphysical. Check wave_time_domain[1][idx_start:]
