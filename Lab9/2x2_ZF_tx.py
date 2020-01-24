# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:22:37 2019

@author: nr29
"""
import SoapySDR
from SoapySDR import * #SOAPY_SDR_constants
import numpy as np
import matplotlib.pyplot as plt
import time
import functions
from scipy import signal

sdr= SoapySDR.Device(dict(driver="iris", serial = "RF3C000044"))
chan = 0
chan2 = 1

sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 40)

sdr.setAntenna(SOAPY_SDR_RX, chan2, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan2, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan2, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan2, 55)

sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 50)

sdr.setAntenna(SOAPY_SDR_TX, chan2, "TRX")
sdr.setSampleRate(SOAPY_SDR_TX, chan2, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan2, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan2, 60)

#------------------------------------------------------------------------------

Fc = 2.5e9               
span = 4
Fs = 20e9
sps = 8
Rsym = Fs/sps

# generating the signal 4qam from mod_gen function file
L = 15000                                # length of binary signal
sig = np.random.randint(0, 2, L)        # generate random msg bits
sig2 = np.random.randint(0,2,L)
binarySig = sig                         # save this for later
binarySig2 = sig2
Original_symbols = functions.QAMmod4(binarySig)
Original_symbols2 = functions.QAMmod4(binarySig2)
sig = Original_symbols
sig2 = Original_symbols2
nSymbs = len(sig)

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


# Upsampling
upSampleFactor = 8
sig = functions.upsampleZeros(sig, upSampleFactor) * 5
sig2 = functions.upsampleZeros(sig2, upSampleFactor) * 5

sig = np.append(LTS, sig)
sig2 = np.append(LTS, sig2)
sig0 = np.append(sig, np.zeros(1000))
sig1 = np.append(sig2, np.zeros(1000))
nsamps = len(sig0)


delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig0.astype(np.complex64)], len(sig0), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig0):
    print("Bad Write 1!!!")
    
sampsRecv0= np.empty([nsamps,2], dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv0[:,0], sampsRecv0[:,1]], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read 1!!!")
    
plt.plot(sampsRecv0[:300,0]) #plots the first 50 (nongarbage samples)
plt.title('Rx1')
plt.show()
plt.plot(sampsRecv0[:300,1]) #plots the first 50 (nongarbage samples)
plt.title('Rx1')
plt.show()

#------------------------------------------------------------------------------

delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [1], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig1.astype(np.complex64)], len(sig1), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig1):
    print("Bad Write 2!!!")
    
sampsRecv1= np.empty([nsamps,2], dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv1[:,0], sampsRecv1[:,1]], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read 2!!!")
    
plt.plot(sampsRecv1[:300,0]) #plots the first 50 (nongarbage samples)
plt.title('Rx2')
plt.show()
plt.plot(sampsRecv1[:300,1]) #plots the first 50 (nongarbage samples)
plt.title('Rx2')
plt.show()

#------------------------------------------------------------------------------

cc = np.correlate(sampsRecv0[0:15000,0],LTS_single,'full')
plt.plot(abs(cc[0:500]))
plt.title('Autocorrelation uwu')
plt.show()
secondPeak = functions.secondPeakLTS(abs(cc))
endLTS00 = secondPeak+1
cc = np.correlate(sampsRecv0[0:15000,1],LTS_single,'full')
secondPeak = functions.secondPeakLTS(abs(cc))
endLTS01 = secondPeak+1
cc = np.correlate(sampsRecv1[0:15000,0],LTS_single,'full')
secondPeak = functions.secondPeakLTS(abs(cc))
endLTS10 = secondPeak+1
plt.plot(abs(cc[0:500]))
plt.title('Autocorrelation uwu')
plt.show()
cc = np.correlate(sampsRecv1[0:15000,1],LTS_single,'full')
secondPeak = functions.secondPeakLTS(abs(cc))
endLTS11 = secondPeak+1

#------------------------------------------------------------------------------

LTSrec00 = np.fft.fft( (sampsRecv0[endLTS00-L:endLTS00,0] + sampsRecv0[endLTS00-2*L:endLTS00-L,0])/2)
LTSrec00 = np.fft.fftshift(LTSrec00)
LTSrec01 = np.fft.fft( (sampsRecv0[endLTS01-L:endLTS01,1] + sampsRecv0[endLTS01-2*L:endLTS01-L,1])/2)
LTSrec01 = np.fft.fftshift(LTSrec01)
LTSrec10 = np.fft.fft( (sampsRecv1[endLTS10-L:endLTS10,0] + sampsRecv1[endLTS10-2*L:endLTS10-L,0])/2)
LTSrec10 = np.fft.fftshift(LTSrec10)
LTSrec11 = np.fft.fft( (sampsRecv1[endLTS11-L:endLTS11,1] + sampsRecv1[endLTS11-2*L:endLTS11-L,1])/2)
LTSrec11 = np.fft.fftshift(LTSrec11)
LTStrans = np.fft.fft(LTS_single)
LTStrans = np.fft.fftshift(LTStrans)
chanEstimate00 = np.multiply(LTSrec00,LTStrans)
chanEstimate01 = np.multiply(LTSrec01,LTStrans)
chanEstimate10 = np.multiply(LTSrec10,LTStrans)
chanEstimate11 = np.multiply(LTSrec11,LTStrans)

# calculate the narrowband channel estimates (average the wideband to find one complex number)
chan00 = np.mean(chanEstimate00)
chan01 = np.mean(chanEstimate01)
chan10 = np.mean(chanEstimate10)
chan11 = np.mean(chanEstimate11)
chan = np.matrix([chan00, chan01], [chan10, chan11])
ZFBF_rec = np.linalg.pinv(chan)

sigMat = np.matrix([sampsRecv0[:,0], sampsRecv0[;,1], [sampsRecv1[:,0], sampsRecv1[:,1]]])

#------------------------------------------------------------------------------
sigDecDown00 = signal.decimate(sampsRecv0[endLTS00:endLTS00+nSymbs*sps,0], upSampleFactor, ftype='fir')
sigDecDown01 = signal.decimate(sampsRecv0[endLTS01:endLTS01+nSymbs*sps,1], upSampleFactor, ftype='fir')
sigDecDown10 = signal.decimate(sampsRecv1[endLTS10:endLTS10+nSymbs*sps,0], upSampleFactor, ftype='fir')
sigDecDown11 = signal.decimate(sampsRecv1[endLTS11:endLTS11+nSymbs*sps,1], upSampleFactor, ftype='fir')

plt.scatter(np.real(sigDecDown00),np.imag(sigDecDown00))
plt.title('Received Constellation Signal 11 vs Original Symbols')
#plt.scatter(np.real(Original_symbols),np.imag(Original_symbols))
plt.show()
plt.scatter(np.real(sigDecDown01),np.imag(sigDecDown01))
plt.title('Received Constellation Signal 12 vs Original Symbols')
#plt.scatter(np.real(Original_symbols2),np.imag(Original_symbols2))
plt.show()
plt.scatter(np.real(sigDecDown10),np.imag(sigDecDown10))
plt.title('Received Constellation Signal 21 vs Original Symbols')
#plt.scatter(np.real(Original_symbols),np.imag(Original_symbols))
plt.show()
plt.scatter(np.real(sigDecDown11),np.imag(sigDecDown11))
plt.title('Received Constellation Signal 22 vs Original Symbols')
#plt.scatter(np.real(Original_symbols2),np.imag(Original_symbols2))
plt.show()









