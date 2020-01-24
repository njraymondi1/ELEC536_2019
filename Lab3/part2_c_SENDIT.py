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
import lab3_functions

sdr= SoapySDR.Device(dict(driver="iris", serial = "0221"))
chan = 0
sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 15)
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 50)


# generating the signal 4qam from mod_gen function file
L = 10000                               # length of binary signal
sig = np.random.randint(0, 2, L)        # generate random msg bits
symbols = lab3_functions.gen_symbols(sig,('qam',4))
symbols = lab3_functions.cpx_awgn(symbols, 0, 0.05)
plt.scatter(np.real(symbols),np.imag(symbols))
plt.title('4QAM Transmitted Signal Constellation')
plt.show()
sig = symbols

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
    
plt.plot(sampsRecv[70:5000]) #plots the first 50 (nongarbage samples)
plt.title('Rx')
plt.show()

# plotting the constellation diagram
iSamps = np.real(sampsRecv[70:3000])#[70:])
qSamps = np.imag(sampsRecv[70:3000])#[70:])
plt.scatter(iSamps,qSamps)
plt.title('Received Signal Constellation')
plt.show()

# plotting the fft
fftData = np.fft.fft(sampsRecv[70:3000])
plt.plot(abs(fftData))
plt.title('FFT Recevied Signal')
plt.show()


















