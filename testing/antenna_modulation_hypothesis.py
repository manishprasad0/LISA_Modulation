import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from lisatools.utils.constants import YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD
from testing.antenna_pattern_utils import *


DAY_SI = 24*3600
Tobs = YRSID_SI
dt = 5.
N = int(Tobs / dt)
Tobs = N * dt
freq = np.fft.rfftfreq(N,dt)
freq[0] = freq[1] # have to do this as freq[0] = 0 causes issues with the waveform generator
wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))


import time
t_start = time.time()

m1 = 1e5
m2 = 2e5
chi1z = 0.1
chi2z = 0.7
dist = 2 * 1e9 * PC_SI
phi_ref = 1.0
f_ref = 0.0
inc = 2*np.pi/3  #2*np.pi/3.
lam = 0
beta= 0
psi = 3*np.pi/8
t_c = Tobs - 5*DAY_SI # chose to have the merger at 5 days before the end of the observation time

modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

wave_freq_domain = wave_gen(m1, m2, chi1z, chi2z,
                            dist, phi_ref, f_ref, inc, lam,
                            beta, psi, t_c, 
                            freqs=freq, modes=modes, 
                            direct=False, fill=True, squeeze=True, length=1024)[0]

wave_time_domain = np.fft.irfft(wave_freq_domain, axis=-1)

f_stft_noiseless, t_stft_noiseless, zxx_noiseless = sp.signal.stft(wave_time_domain[1], fs=1/dt, nperseg=15000, noverlap=0)         # In wave_time_domain: 0 is A, 1 is E, 2 is T

t_months = t_stft_noiseless/(24*3600)/30.45428241

max_zxx = []
max_zxx = np.max(np.abs(zxx_noiseless), axis=0)
#max_zxx_normalized = max_zxx/np.max(np.array(max_zxx))

print(f"Max ampliture per STFT time bin is in: {max_zxx}")

plt.figure(figsize=(10, 6))

plt.plot(t_months, max_zxx)
plt.ylabel('Maximum Normalized Amplitude per Time Bin')
plt.xlabel('Time [months]')
plt.yscale('log')
plt.grid(True, which='both', alpha=0.4)
plt.xlim(t_months[0], t_months[-1])
plt.xticks(np.arange(0, 13, 1))
plt.tight_layout()
plt.show()