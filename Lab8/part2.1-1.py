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

Fc = 2.5e9               
span = 4
Fs = 20e9
sps = 8
Rsym = Fs/sps

# generating the signal 4qam from mod_gen function file
L = 1000                                # length of binary signal
sig = np.random.randint(0, 2, L)        # generate random msg bits
binarySig = sig                         # save this for later
Original_symbols = functions.QAMmod4(binarySig)
sig = Original_symbols
nSymbs = len(sig)
nsamps = len(sig) # this is now nSymbs

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

sig = np.append(LTS, sig)
sig = np.append(sig, np.zeros(200))
nsamps = len(sig)


# Channel 1--------------------------------------------------------------------
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
#------------------------------------------------------------------------------

sampsRecv = sig*(1+1j)                 #comment this to remove transmission channel
plt.plot(sampsRecv[0:500])
plt.title('Recevied Signal 1 - No operations')
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
#------------------------------------------------------------------------------


# get channel estimates before downsampling----------------------------------
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


# calculate the narrowband channel estimates (average the wideband to find one complex number)
chan1 = np.mean(chanEstimate)/3.25

# equalize by dividing received signal by channel estimate
eqSig1 = sampsRecv[endLTS:endLTS+nSymbs*sps]/chan1

#------------------------------------------------------------------------------

# Downsample our equalized signal by a factor of 8-----------------------------
sigDecDown = signal.decimate(sampsRecv[endLTS:endLTS+nSymbs*sps], upSampleFactor, ftype='fir')
#sigDecDown = signal.decimate(eqSig1, upSampleFactor, ftype='fir')
plt.scatter(np.real(sigDecDown),np.imag(sigDecDown))
plt.title('Received Constellation Signal 1')
plt.show()
plt.scatter(np.real(Original_symbols),np.imag(Original_symbols))
plt.title('Modulated Symbol Constellation')
plt.show()
#------------------------------------------------------------------------------

# recevier
sumSig = sigDecDown

# demodulation
[sig2bits,sig2symbs] = functions.QAMdemod4(sumSig)
BERsig = functions.BERcalc(binarySig, sig2bits)
print('BER')
print(BERsig)

plt.scatter(np.real(sumSig),np.imag(sumSig))
plt.title('Recovered Constellation')
plt.show()

# =============================================================================
# for qam, hard in zip(recoveredSymbols, sig2symbs):
#     plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
#     plt.plot(sig2symbs.real, sig2symbs.imag, 'ro')
# plt.title('Recovered Constellation after Alamouti Post Processing')
# plt.show()
# =============================================================================











