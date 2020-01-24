# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:18:55 2019

@author: nr29
"""

import SoapySDR
from SoapySDR import * #SOAPY_SDR_constants
import numpy as np
import matplotlib.pyplot as plt
import functions
from scipy import signal

sdr= SoapySDR.Device(dict(driver="iris", serial = "RF3C000063"))
chan = 0
sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 20e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 20)
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
Original_symbols = functions.QAMmod4(binarySig)
[symbols,symbols2] = functions.alamoutiv2(Original_symbols)
#symbols = functions.dqpskMod(sig)
sig = symbols
nSymbs = len(sig)
nsamps = len(sig) # this is now nSymbs

#rCosLength = sps*span
#rCos = functions.rrcosFilter(rCosLength, 0.25)
#shaped = np.convolve(sig, rCos)
#shaped2 = np.convolve(symbols2, rCos)

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
sig = functions.upsampleZeros(sig, upSampleFactor)
sig2 = functions.upsampleZeros(symbols2, upSampleFactor)

# Appending the LTS to the start of the signal after Upsampling and Upconversion
#sig = np.append(LTS, sigUp)
sig = np.append(LTS, sig)
sig2 = np.append(LTS, sig2)
#sig = np.concatenate((sig, sig))                # MAKE TWO COPIES OF THE SIGNAL
sig = np.append(sig, np.zeros(200))
sig2 = np.append(sig2, np.zeros(200))
nsamps = len(sig)


#UNCOMMENT THIS TO ACTUALLY SEND THE DATA
# Channel 1--------------------------------------------------------------------
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
#------------------------------------------------------------------------------
# Channel 2--------------------------------------------------------------------
delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig.astype(np.complex64)], len(sig2), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig):
    print("Bad Write!!!")
#ts= int(sdr.getHardwareTime() + delay)
sampsRecv2= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv2], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
#------------------------------------------------------------------------------
# Channel 3--------------------------------------------------------------------
delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig2.astype(np.complex64)], len(sig2), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig2):
    print("Bad Write!!!")
#ts= int(sdr.getHardwareTime() + delay)
sampsRecv3= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv3], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
#------------------------------------------------------------------------------
# Channel 4--------------------------------------------------------------------
delay = 10e6
txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(txStream)
sr= sdr.writeStream(txStream, [sig2.astype(np.complex64)], len(sig2), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig2):
    print("Bad Write!!!")
#ts= int(sdr.getHardwareTime() + delay)
sampsRecv4= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdr.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdr.readStream(rxStream, [sampsRecv4], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
#------------------------------------------------------------------------------    
    
    
    
# =============================================================================
# sampsRecv = sig # uncomment this to remove channel
# sampsRecv2 = sig
# sampsRecv3 = sig2
# sampsRecv4 = sig2
# =============================================================================


plt.plot(sampsRecv[0:500])#:5000]) #plots the first 50 (nongarbage samples)
plt.title('Recevied Signal - No operations')
plt.show()

#------------------------------------------------------------------------------
# LTS Stuff
cc = np.correlate(sampsRecv,LTS_single,'full')
#plt.plot(abs(cc[0:500]))    
plt.plot(abs(cc))           
plt.title('Cross Correlation LTS - Close Up')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()
secondPeak = functions.secondPeakLTS(abs(cc))
endLTS = secondPeak+1
print(endLTS)


cc2 = np.correlate(sampsRecv2,LTS_single,'full')  
secondPeak2 = functions.secondPeakLTS(abs(cc2))
endLTS2 = secondPeak2+1
cc3 = np.correlate(sampsRecv3,LTS_single,'full')  
secondPeak3 = functions.secondPeakLTS(abs(cc3))
endLTS3 = secondPeak3+1
cc4 = np.correlate(sampsRecv4,LTS_single,'full')  
secondPeak4 = functions.secondPeakLTS(abs(cc4))
endLTS4 = secondPeak4+1
#------------------------------------------------------------------------------


# get channel estimates before downconversion----------------------------------
# start with end of CP, divide received LTSes by transmitted per sample
LTSrec = np.fft.fft( (sampsRecv[endLTS-L:endLTS] + sampsRecv[endLTS-2*L:endLTS-L])/2)
LTSrec = np.fft.fftshift(LTSrec)
LTStrans = np.fft.fft(LTS_single)
LTStrans = np.fft.fftshift(LTStrans)
chanEstimate = np.multiply(LTSrec,LTStrans)

plt.plot(abs(chanEstimate))
plt.title('Magnitude of Wideband Channel 1 Estimate')
plt.xlabel('Frequency Domain')
plt.show()
plt.plot(np.angle(chanEstimate))
plt.title('Phase of Wideband Channel 1 Estimate')
plt.xlabel('Frequency Domain')
plt.show()

LTSrec2 = np.fft.fft( (sampsRecv2[endLTS2-L:endLTS2] + sampsRecv2[endLTS2-2*L:endLTS2-L])/2)
LTSrec2 = np.fft.fftshift(LTSrec2)
LTStrans2 = np.fft.fft(LTS_single)
LTStrans2 = np.fft.fftshift(LTStrans2)
chanEstimate2 = np.multiply(LTSrec2,LTStrans2)
LTSrec3 = np.fft.fft( (sampsRecv3[endLTS3-L:endLTS3] + sampsRecv3[endLTS3-2*L:endLTS3-L])/2)
LTSrec3 = np.fft.fftshift(LTSrec3)
LTStrans3 = np.fft.fft(LTS_single)
LTStrans3 = np.fft.fftshift(LTStrans3)
chanEstimate3 = np.multiply(LTSrec3,LTStrans3)
LTSrec4 = np.fft.fft( (sampsRecv4[endLTS4-L:endLTS4] + sampsRecv4[endLTS4-2*L:endLTS4-L])/2)
LTSrec4 = np.fft.fftshift(LTSrec4)
LTStrans4 = np.fft.fft(LTS_single)
LTStrans4 = np.fft.fftshift(LTStrans4)
chanEstimate4 = np.multiply(LTSrec4,LTStrans4)


# calculate the narrowband channel estimates (average the wideband to find one complex number)
chan1 = np.mean(chanEstimate[10:55])
chan2 = np.mean(chanEstimate2[10:55])
chan3 = np.mean(chanEstimate3[10:55])
chan4 = np.mean(chanEstimate4[10:55])

# equalize by dividing received signal by channel estimate
#eqSig1 = sampsRecv[endLTS:endLTS+nSymbs*sps]/chan1
#eqSig2 = sampsRecv2[endLTS2:endLTS2+nSymbs*sps]/chan2
#------------------------------------------------------------------------------


# Downsample our equalized signal by a factor of 8-----------------------------
sigDecDown = signal.decimate(sampsRecv[endLTS:endLTS+nSymbs*sps], upSampleFactor, ftype='fir')
#sigDecDown = signal.decimate(eqSig1, upSampleFactor, ftype='fir')
# =============================================================================
# plt.scatter(np.real(sigDecDown),np.imag(sigDecDown))
# plt.title('Received Constellation Signal 1')
# plt.show()
# =============================================================================
sigDecDown2 = signal.decimate(sampsRecv2[endLTS2:endLTS2+nSymbs*sps], upSampleFactor, ftype='fir')
#sigDecDown2 = signal.decimate(eqSig2, upSampleFactor, ftype='fir')
# =============================================================================
# plt.scatter(np.real(sigDecDown2),np.imag(sigDecDown2))
# plt.title('Received Constellation Signal 2')
# plt.show()
# plt.scatter(np.real(Original_symbols),np.imag(Original_symbols))
# plt.title('Modulated Symbol Constellation')
# plt.show()
# =============================================================================
sigDecDown3 = signal.decimate(sampsRecv3[endLTS3:endLTS3+nSymbs*sps], upSampleFactor, ftype='fir')
sigDecDown4 = signal.decimate(sampsRecv4[endLTS4:endLTS4+nSymbs*sps], upSampleFactor, ftype='fir')
#------------------------------------------------------------------------------

# sigDecDown [s1 -s2* s3 -s4* ...] where s = hx
# sigDecDown2 [s2 s1* s4 s3* ...]

#sigDecDown4 = np.zeros(len(sigDecDown))
#chan4 = 0

# LEGEND sig = 1,2, sig2 = 3,4
# chan1 = h11, chan2 = h21, chan3 = h12, chan4 = h22

# recevier
sumSig = sigDecDown + sigDecDown3   #len = symbols
# this has form [s1+s2, -s2*+s1*, s3+s4, -s4*+s3,...] 
sumSig2 = sigDecDown2 + sigDecDown4
sumSig = sumSig + sumSig2

plt.scatter(np.real(sumSig),np.imag(sumSig))
plt.title('Received Symbols Before Alamouti Post')
plt.show()

partitioned = sumSig.reshape(-1,2)
partitioned2 = sumSig2.reshape(-1,2)
# these have form [s1+s2, -s2*+s1*]
#               [s3+s4, -s4*+s3*]...

# receieve antenna 1
recoveredSymbols = np.array(np.zeros((len(partitioned),2)),dtype=np.complex128)
for i in range(len(partitioned)):
    recoveredSymbols[i][0] = np.multiply(np.conj(chan1), partitioned[i][0]) + np.multiply(chan3, np.conj(partitioned[i][1])) + np.multiply(np.conj(chan2), partitioned2[i][0]) + np.multiply(chan4, np.conj(partitioned2[i][1]))
    recoveredSymbols[i][1] = np.multiply(np.conj(chan3), partitioned[i][0]) - np.multiply(chan1, np.conj(partitioned[i][1])) + np.multiply(np.conj(chan4), partitioned2[i][0]) - np.multiply(chan2, np.conj(partitioned2[i][1]))
recoveredSymbols = recoveredSymbols.reshape(len(symbols),1)

# receive antenna 2
# =============================================================================
# recoveredSymbols2 = np.array(np.zeros((len(partitioned2),2)),dtype=np.complex128)
# for i in range(len(partitioned2)):
#     recoveredSymbols2[i][0] = np.multiply(np.conj(chan2), partitioned2[i][0]) + np.multiply(chan4, np.conj(partitioned2[i][1]))
#     recoveredSymbols2[i][1] = np.multiply(np.conj(chan4), partitioned2[i][0]) - np.multiply(chan2, np.conj(partitioned2[i][1]))
# recoveredSymbols2 = recoveredSymbols2.reshape(len(symbols),1)
# =============================================================================

recovered = recoveredSymbols

plt.scatter(np.real(recovered),np.imag(recovered))
plt.title('Recovered Constellation after Alamouti - 2 Antennas')
plt.show()

[sig2bits,sig2symbs] = functions.QAMdemod4(recovered)
BERsig = functions.BERcalc(binarySig, sig2bits)
print('Post-Alamouti BER - 2 Antenna Receiver ')
print(BERsig)


















