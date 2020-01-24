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
sdr.setGain(SOAPY_SDR_RX, chan, 25)
sdr.setSampleRate(SOAPY_SDR_TX, chan, 20e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 25)

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
sr= sdr.writeStream(txStream, [sig2.astype(np.complex64)], len(sig2), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig2):
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
    
    
    
sampsRecv = sig # uncomment this to remove channel
sampsRecv2 = sig2
plt.plot(sampsRecv[0:500])#:5000]) #plots the first 50 (nongarbage samples)
plt.title('Recevied Signal 1 - No operations')
plt.show()

# SIMULATE ONE OF THE ENDS TERMINATED 
plt.plot(sampsRecv2[0:500])#:5000]) #plots the first 50 (nongarbage samples)
plt.title('Recevied Signal 2 - No operations')
plt.show()

#------------------------------------------------------------------------------
# LTS Stuff
cc = np.correlate(sampsRecv,LTS_single,'full')
#plt.plot(abs(cc[0:500]))    
plt.plot(abs(cc))           
plt.title('Cross Correlation LTS1')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()
secondPeak = functions.secondPeakLTS(abs(cc))
endLTS = secondPeak+1
print(endLTS)

# LTS Stuff
cc2 = np.correlate(sampsRecv2,LTS_single,'full')
#plt.plot(abs(cc[0:500]))    
plt.plot(abs(cc2))           
plt.title('Cross Correlation LTS2')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()
secondPeak2 = functions.secondPeakLTS(abs(cc2))
endLTS2 = secondPeak2+1
print(endLTS2)
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

plt.plot(abs(chanEstimate2))
plt.title('Magnitude of Wideband Channel 2 Estimate')
plt.xlabel('Frequency Domain')
plt.show()
plt.plot(np.angle(chanEstimate2))
plt.title('Phase of Wideband Channel 2 Estimate')
plt.xlabel('Frequency Domain')
plt.show()

# calculate the narrowband channel estimates (average the wideband to find one complex number)
chan1 = np.mean(chanEstimate[10:50])
chan2 = np.mean(chanEstimate2[10:50])
#chan1 = np.mean(np.fft.ifft(chanEstimate))
#chan2 = np.mean(np.fft.ifft(chanEstimate2))

# equalize by dividing received signal by channel estimate
eqSig1 = sampsRecv[endLTS:endLTS+nSymbs*sps]/chan1
eqSig2 = sampsRecv2[endLTS2:endLTS2+nSymbs*sps]/chan2
#------------------------------------------------------------------------------


# Downsample our equalized signal by a factor of 8-----------------------------
sigDecDown = signal.decimate(sampsRecv[endLTS:endLTS+nSymbs*sps], upSampleFactor, ftype='fir')
#sigDecDown = signal.decimate(eqSig1, upSampleFactor, ftype='fir')
plt.scatter(np.real(sigDecDown),np.imag(sigDecDown))
plt.title('Received Constellation Signal 1')
plt.show()
sigDecDown2 = signal.decimate(sampsRecv2[endLTS2:endLTS2+nSymbs*sps], upSampleFactor, ftype='fir')
#sigDecDown2 = signal.decimate(eqSig2, upSampleFactor, ftype='fir')
plt.scatter(np.real(sigDecDown2),np.imag(sigDecDown2))
plt.title('Received Constellation Signal 2')
plt.show()
plt.scatter(np.real(Original_symbols),np.imag(Original_symbols))
plt.title('Modulated Symbol Constellation')
plt.show()
#------------------------------------------------------------------------------

# sigDecDown [s1 -s2* s3 -s4* ...] where s = hx
# sigDecDown2 [s2 s1* s4 s3* ...]

# simulate one of the ends being terminated
sigDecDown2 = np.zeros(len(sigDecDown))
chan2 = 0

# recevier
sumSig = sigDecDown + sigDecDown2
# this has form [s1+s2, -s2*+s1*, s3+s4, -s4*+s3,...] 
plt.scatter(np.real(sumSig),np.imag(sumSig))
plt.title('Received Symbols Before Alamouti Post')
plt.show()

partitioned = sumSig.reshape(-1,2)
# this has form [s1+s2, -s2*+s1*]
#               [s3+s4, -s4*+s3*]...

recoveredSymbols = np.array(np.zeros((len(partitioned),2)),dtype=np.complex128)
for i in range(len(partitioned)):
    recoveredSymbols[i][0] = (np.conj(chan1) * partitioned[i][0]) + (chan2 * np.conj(partitioned[i][1]))
    recoveredSymbols[i][1] = (np.conj(chan2) * partitioned[i][0]) - (chan1 * np.conj(partitioned[i][1]))
recoveredSymbols = recoveredSymbols.reshape(len(symbols),1)


# demodulation
[sig2bits,sig2symbs] = functions.QAMdemod4(recoveredSymbols)
BERsig = functions.BERcalc(binarySig, sig2bits)
print('Post-Alamouti BER')
print(BERsig)

plt.scatter(np.real(recoveredSymbols),np.imag(recoveredSymbols))
plt.title('Recovered Constellation after Alamouti Post Processing')
plt.show()

# =============================================================================
# for qam, hard in zip(recoveredSymbols, sig2symbs):
#     plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
#     plt.plot(sig2symbs.real, sig2symbs.imag, 'ro')
# plt.title('Recovered Constellation after Alamouti Post Processing')
# plt.show()
# =============================================================================











