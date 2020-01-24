# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:57:43 2019

@author: nr29
"""

import SoapySDR
from SoapySDR import * #SOAPY_SDR_constants
import numpy as np
import matplotlib.pyplot as plt
import functions
from scipy import signal

sdr= SoapySDR.Device(dict(driver="iris", serial = "RF3C000044"))
chan = 0
sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 20e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 40)
sdr.setSampleRate(SOAPY_SDR_TX, chan, 20e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 40)

Fc = 2.5e9               
span = 4
Fs = 20e9
sps = 8
Rsym = Fs/sps

# generating the signal 4qam from mod_gen function file
L = 1000                          # length of binary signal
sig = np.random.randint(0, 2, L)        # generate random msg bits
binarySig = sig                 # save this for later
#symbols = functions.gen_symbols(sig,('qam',4))
symbols = functions.dqpskMod(sig)
sig = symbols
nSymbs = len(sig)
nsamps = len(sig) # this is now nSymbs

rCosLength = sps*span
rCos = functions.rrcosFilter(rCosLength, 0.25)
shaped = np.convolve(sig, rCos)

# Generating LTS
LTS = [0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,
       1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,
       1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,
       1,1,1,0,0,0,0,0]
LTS = np.fft.ifftshift(LTS)              # perform iFFTshift
LTS_single = np.fft.ifft(LTS)              # take ifft
L = len(LTS)
CP = LTS_single[L-32:]
lenCP = len(CP)
LTS = np.concatenate((CP,LTS_single,LTS_single))*4
LTS = np.array(LTS)
lenLTS = len(LTS)

# MAKE OTHER LTSes with phase and amplitudes to simulate multipath
LTS_2_off = np.multiply(np.insert(0.7*LTS[0:len(LTS)-2],0,(0,0)),(1+1j))                    # 0.7 mag, phase = 45 deg
LTS_5_off = np.multiply(np.insert(0.5*LTS[0:len(LTS)-5],0,(0,0,0,0,0)),(np.sqrt(3)/2+0.5j)) # 0.5 mag, phase = 30 deg
LTS_7_off = np.multiply(np.insert(0.35*LTS[0:len(LTS)-7],0,(0,0,0,0,0,0,0)),(0.97+0.35j))   # 0.35 mag, phase = 10 deg
LTSsum = LTS + LTS_2_off + LTS_5_off + LTS_7_off
LTS = LTSsum

# Upsampling
upSampleFactor = 8
sig = functions.upsampleZeros(sig, upSampleFactor)
shapedUp = np.convolve(sig, rCos)
shapedUp = shapedUp[0:len(sig)]

plt.stem(sig[0:50*upSampleFactor])
plt.title('4QAM Modulated Data Signal Symbols - UPSAMPLED')
plt.hold(True)
plt.plot(shapedUp[int(rCosLength/2):50*upSampleFactor+int(rCosLength/2)], color = 'red')
plt.title('Conv of Mod Data and rCof Coefficients - UPSAMPLED DATA')
plt.show()

# Upconverting
f,Px = signal.welch(shapedUp, Fs, return_onesided=False, detrend=False, scaling='spectrum')
plt.semilogy(f,Px)
plt.title('PSD of Modulated 8x Upsampled Filtered Baseband Signal')
plt.show()

sigUp = functions.upconversionv2(shapedUp, Fc, Fs, sps, nSymbs)
f,Px = signal.welch(sigUp, Fs, return_onesided=False, detrend=False, scaling='spectrum')
plt.semilogy(f,Px)
plt.title('PSD of Modulated 8x Upsampled Filtered Upconverted 2.5GHz Signal')
plt.show()

# Appending the LTS to the start of the signal after Upsampling and Upconversion
sig = np.append(LTS, sigUp)
sig = np.append(sig, np.zeros(200))
#sig = np.pad(sig, 100, 'constant')
nsamps = len(sig)

# =============================================================================
# UNCOMMENT THIS TO SEE THE DOWNCONVRETED PSD BEFORE TRANSMISSION
# sigDown = functions.downconversionv2(sigUp, Fc, Fs, sps, nSymbs)
# f,Px = signal.welch(sigDown, Fs, return_onesided=False, detrend=False, scaling='spectrum')
# plt.semilogy(f,Px)
# plt.title('PSD of Modulated 8x Upsampled Filtered Up/Transmitted/Down Converted Baseband Signal')
# plt.show()
# =============================================================================


#UNCOMMENT THIS TO ACTUALLY SEND THE DATA
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
    
    
sampsRecv = sig # uncomment this to remove channel
plt.plot(sampsRecv[0:500])#:5000]) #plots the first 50 (nongarbage samples)
plt.title('Recevied Signal - No operations')
plt.show()

# LTS Stuff
#cc = np.correlate(LTS_single,sampsRecv[0:lenLTS],"full")
cc = np.correlate(sampsRecv,LTS_single,'full')
#plt.plot(abs(cc))             
#plt.title('Cross Correlation LTS')
#plt.xlabel('Lag')
#plt.ylabel('Cross Correlation')
#plt.show()
plt.plot(abs(cc[0:200]))             
plt.title('Cross Correlation LTS - Close Up')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()
plt.plot(abs(cc[150:175]))             
plt.title('Cross Correlation LTS - Closer Up')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()
secondPeak = functions.secondPeakLTS(abs(cc))
#print(secondPeak)
endLTS = secondPeak+1           # hotfix i don't know why but verified
print(secondPeak)

sampsRecv = functions.removeDC(sampsRecv)

# get channel estimates before downconversion
# start with end of CP, divide received LTSes by transmitted per sample
LTSrec = np.fft.fft(sampsRecv[endLTS-2*L:endLTS])
LTStrans = np.fft.fft(LTS[lenCP:])
chanEstimate = np.multiply(LTSrec,LTStrans) 

plt.plot(LTSrec)
plt.title('Recevied LTS')
plt.show()
plt.plot(LTStrans)
plt.title('Transmitted LTS')
plt.show()

plt.plot(abs(chanEstimate))
plt.title('Magnitude of Channel Estimate')
plt.show()
plt.plot(np.angle(chanEstimate))
plt.title('Phase of Channel Estimate')
plt.show()

# Downconvert
sigDown = functions.downconversionv2(sampsRecv[endLTS:endLTS+nSymbs*sps], Fc, Fs, sps, nSymbs)
f,Px = signal.welch(sigDown, Fs, return_onesided=False, detrend=False, scaling='spectrum')
plt.semilogy(f,Px)
plt.title('PSD of Modulated 8x Upsampled Filtered Up/Transmitted/Down Converted Baseband Signal')
plt.show()

# Downsample by a factor of 8
sigDecDown = signal.decimate(sigDown, upSampleFactor, ftype='fir')

# plotting constellations
# plotting the constellation diagram
# =============================================================================
# iSamps = np.real(sigDecDown)#[70:3000])#[70:])
# qSamps = np.imag(sigDecDown)#[70:3000])#[70:])
# plt.scatter(iSamps,qSamps)
# plt.title('Received Signal Constellation')
# plt.show()
# 
# iSamps = np.real(symbols)#[70:3000])#[70:])
# qSamps = np.imag(symbols)#[70:3000])#[70:])
# plt.scatter(iSamps,qSamps)
# plt.title('Transmitted Signal Constellation')
# plt.show()
# =============================================================================