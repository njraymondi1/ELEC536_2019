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
sdr.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 25)
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 40)

# generating the signal
nsamps = 78000*2
nsamps_pad = 100
s_freq = 10e6
Ts = 10/nsamps
s_time_vals = np.array(np.arange(0,nsamps)).transpose()*Ts
sig = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.5
#sig_pad = np.concatenate((np.zeros(nsamps_pad), sig, np.zeros(nsamps_pad)))
plt.plot(sig)
plt.title('Tx')
plt.show()

delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig.astype(np.complex64)], len(sig), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig):
    print("Bad Write!!!")
    
# ts= sdr.getHardwareTime() + delay #give us delay ns to set everything up.
sampsRecv= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
    
# plotting recevied signal    
plt.plot(sampsRecv)
plt.title('Rx')
plt.show()

# plotting the constellation diagram
iSamps = np.real(sampsRecv)
qSamps = np.imag(sampsRecv)
plt.scatter(iSamps,qSamps)
plt.title('Scatterplot')
plt.show()

# plotting the fft
fftData = np.fft.fft(sampsRecv)
plt.plot(abs(fftData))
plt.title('FFT Recevied Signal')
plt.show()


















