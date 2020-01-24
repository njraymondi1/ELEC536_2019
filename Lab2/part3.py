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
from IPython.display import display

sdrTx= SoapySDR.Device(dict(driver="iris", serial = "RFC000031"))
sdrRx= SoapySDR.Device(dict(driver="iris", serial = "0221"))
chan = 0
sdrTx.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdrTx.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdrTx.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdrTx.setGain(SOAPY_SDR_TX, chan, 35)
sdrRx.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdrRx.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdrRx.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdrRx.setGain(SOAPY_SDR_RX, chan, 50)



# generating the signal
#   802.11 LTS
sigLTS = [0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,
       1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,
       1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,
       1,1,1,0,0,0,0,0]
plt.plot(sigLTS)
plt.title('LTS Signal Digital Domain')
plt.show()
sigIFFT = np.fft.ifft(sigLTS)
sig = np.fft.ifftshift(sigIFFT)          # perform iFFT and ifftshift 
sig = np.append([sig], [sig])            # repeat twice (1)
sig = np.append([sig], [sig])            # repeat twice (2)
LTSlen = len(sig)
sig = np.insert(sig, 1, sig[L-16:])      # CP, attach last 16 samples onto the begining of the signal
sig = np.array(sig)
plt.plot(sig)
plt.title('Tx Signal - iFFT(LTS)')
plt.show()

# generating the complex sinusoid 
f = 500e3
nsamps = 1000
fs = 10e6
n = np.arange(nsamps)
T = 1.0/fs
t = nsamps*T
phi = 0
sinusoid = np.exp(1j*(2*np.pi*f*n*T + phi))*0.5
xim = np.imag(sig)
plt.plot(sinusoid[0:50])
plt.title('Tx - Complex Sinudoid')
plt.show()

# putting sinusoid after LTS
sig = np.append([sig],[sinusoid])
sigPad = np.pad(sig, 100, 'constant')    # zero padding was required to get it thru
nsamps = len(sigPad)
n = np.arange(nsamps)
plt.plot(sig)
plt.title('Tx Signal')
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
    
sigR = sampsRecv[167:167+len(sig)]  # only use good samples
plt.plot(sigR) 
plt.title('Rx')
plt.show()

#sig = (sig-np.mean(sig))/np.std(sig)
#sigR = (sigR-np.mean(sigR))/np.std(sigR)

# perform cross-correlation on LTS
cc = np.correlate(sig,sigR,"full")
#m = np.argmax(abs(cc))
plt.plot(abs(cc))
plt.title('Cross Correlation LTS + Complex Sinusoid')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()

# find max sample of cc
m = np.argmax(abs(cc))
print(m)

sinusoidRec = sigR[LTSlen+16:]  # LTS + 16 to account for CP
sinNormalized = sinusoidRec/sinusoid
plt.plot(abs(sinNormalized[0:200])) 
plt.title('Normalized Sinusoid Magnitude')
plt.show()
plt.plot(np.angle(sinNormalized[0:200], deg=True)) 
plt.title('Normalized Sinusoid Angle')
plt.show()

ltsR = sigR[16:64+16]  # start at 16 to account for CP
fftData = np.fft.fft(ltsR)
plt.plot(fftData) 
plt.title('FFT of Received LTS')
plt.show()



#calculate the phase difference using the peaks of the cc function
lag1 = np.angle(cc[207], deg=True)
lag2 = np.angle(cc[271], deg=True)
#phase_difference = lag1-lag2
#print(phase_difference)
lags = np.angle(cc, deg=True)
a = np.array(range(64))
for i in range(63):
    a[i] = lags[i] - lags[i+64]
    print(a[i])
CFO = np.mean(a)


