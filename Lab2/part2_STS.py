# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:36:56 2019

@author: nr29
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:20:31 2019

@author: nr29
"""
import SoapySDR
from SoapySDR import * #SOAPY_SDR_constants
import numpy as np
import matplotlib.pyplot as plt

sdr= SoapySDR.Device(dict(driver="iris", serial = "0221"))
chan = 0
sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 20)
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 30)


# generating the signal
#   802.11 STS
sig = [0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,
       0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,0,0,0,0,
       -1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,
       1+1j,0,0,0,1+1j,0,0,0,0,0,0,0]
plt.plot(sig)
plt.title('Tx')
plt.show()
sig = np.fft.ifft(sig)                  # perform iFFT
sig = np.append([sig], [sig])           # repeat twice (1)
sig = np.append([sig], [sig])           # repeat twice (2)
L = len(sig)
sig = np.insert(sig, 1, sig[L-16:])     # cyclic prefix, attach last 16 onto the begining of the signal
sig = np.array(sig)
sigPad = np.pad(sig, 100, 'constant')   # zero padding was required to get it thru
nsamps = len(sigPad)
n = np.arange(nsamps)
plt.plot(sig)
plt.title('Tx')
plt.show()

delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sigPad.astype(np.complex64)], len(sigPad), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sigPad):
    print("Bad Write!!!")
    
#ts= int(sdr.getHardwareTime() + delay)
sampsRecv= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
    
sigR = sampsRecv[167:167+len(sig)] # only take the good samples
plt.plot(sigR) 
plt.title('Rx')
plt.show()

plt.acorr(sigR, maxlags=50)
plt.title('Autocorrelation of STS')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# =============================================================================
# fftData = np.fft.ifft(sigR)
# plt.plot(abs(fftData))
# plt.title('FFT Recevied Signal')
# plt.show()
# =============================================================================

