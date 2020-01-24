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

sdr= SoapySDR.Device(dict(driver="iris", serial = "RF3C000044"))
chan = 0
chan2 = 1

sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 40)

sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 40)

sdr.setAntenna(SOAPY_SDR_TX, chan2, "TRX")
sdr.setSampleRate(SOAPY_SDR_TX, chan2, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan2, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan2, 40)


# generating the signal
f = 500e3
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
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig.astype(np.complex64),sig.astype(np.complex64)], len(sig), txFlags, timeNs=ts)
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
    
plt.plot(sampsRecv[70:120]) #plots the first 50 (nongarbage samples)
plt.title('Rx')
plt.show()

# plotting the constellation diagram
iSamps = np.real(sampsRecv[70:])
qSamps = np.imag(sampsRecv[70:])
plt.scatter(iSamps,qSamps)
plt.title('Scatterplot')
plt.show()

# plotting the fft
fftData = np.fft.fft(sampsRecv[70:])
plt.plot(abs(fftData))
plt.title('FFT Recevied Signal')
plt.show()

# received signal strength 
RSS = np.mean(abs(sampsRecv))
print('Received Signal Strength')
print(RSS)
















