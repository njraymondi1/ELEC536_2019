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
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 300e6)
sdr.setGain(SOAPY_SDR_RX, chan, 20)
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 900e6)
sdr.setGain(SOAPY_SDR_TX, chan, 30)


# generating the signal
f = 301e6               # 301 MHz
nsamps = 10000
fs = 10e6
n = np.arange(nsamps)
T = 1.0/fs
t = nsamps*T
phi = 0
sig = np.exp(1j*(2*np.pi*f*n*T + phi))*0.5
xim = np.imag(sig)
plt.plot(sig[0:50])
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
    
#ts= int(sdr.getHardwareTime() + delay)
sampsRecv= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
    
plt.plot(sampsRecv[70:120]) #plots 50 (nongarbage samples)
plt.title('Rx')
plt.show()

# plotting the constellation diagram
iSamps = np.real(sampsRecv[70:])  # only consider non garbage samples
qSamps = np.imag(sampsRecv[70:])  # only consider non garbage samples
plt.scatter(iSamps,qSamps)
plt.title('Scatterplot')
plt.show()

# plotting the fft
fftData = np.fft.fft(sampsRecv[70:])
plt.plot(np.flip(abs(fftData),axis=0))
plt.title('FFT Recevied Signal')
plt.show()

print(np.argmax(fftData))
print(300e6/301e6)

















