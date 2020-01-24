# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:14:46 2019

@author: nr29
"""

# =============================================================================
# import SoapySDR
# from SoapySDR import * #SOAPY_SDR_constants
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import functions
from scipy import signal

# =============================================================================
# sdr= SoapySDR.Device(dict(driver="iris", serial = "0221"))
# chan = 0
# sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
# sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
# sdr.setSampleRate(SOAPY_SDR_RX, chan, 20e6)
# sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
# sdr.setGain(SOAPY_SDR_RX, chan, 25)
# sdr.setSampleRate(SOAPY_SDR_TX, chan, 20e6)
# sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
# sdr.setGain(SOAPY_SDR_TX, chan, 30)
# =============================================================================

Fc = 2.5e9               
span = 4
Fs = 20e9
sps = 8
Rsym = Fs/sps

# generating the signal 4qam from mod_gen function file
L = 10000                          # length of binary signal
sig = np.random.randint(0, 2, L)        # generate random msg bits
symbols = functions.gen_symbols(sig,('qam',4))
######symbols = functions.dqpskMod(sig)
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
#plt.plot(np.abs(sig))
#plt.title('LTS Signal Frequency Domain')
#plt.show()
LTS = np.fft.ifftshift(LTS)              # perform iFFTshift
#plt.plot(sig)
#plt.title('FFTShift of LTS')
#plt.show()
LTS_FFT = np.fft.ifft(LTS)              # take ifft
LTS = np.append([LTS_FFT], [LTS_FFT])            # repeat twice (1)
L = len(LTS)
LTS = np.insert(LTS, -1, LTS[L-32:])    # cyclic prefix, attach last 32 onto the begining of the signal
LTS = np.array(LTS)
lenLTS = len(LTS)
#LTSPad = np.pad(LTS, 100, 'constant')    # zero padding was required to get it thru
#nsamps = len(LTSPad)
#n = np.arange(nsamps)
#plt.plot(sig)
#plt.title('Tx Signal - iFFT(LTS)')
#plt.show()

# Upsampling
upSampleFactor = 8
sig = functions.upsampleZeros(sig, upSampleFactor)
shapedUp = np.convolve(sig, rCos)
shapedUp = shapedUp[0:len(sig)]

plt.stem(sig[0:50*upSampleFactor])
plt.title('4QAM Modulated Data Signal Symbols - UPSAMPLED')
plt.hold(True)
plt.plot(shapedUp[int(rCosLength/2):400+int(rCosLength/2)], color = 'red')
plt.title('Conv of Mod Data and rCof Coefficients - UPSAMPLED DATA')
plt.show()

# Upconverting
sigUp = functions.upconversionv2(shapedUp, Fc, Fs, sps, nSymbs)
#window = signal.get_window('hamming', 512)

f,Px = signal.welch(shapedUp, Fs, return_onesided=False, detrend=False, scaling='spectrum')
plt.semilogy(f,Px)
plt.title('PSD of Modulated 8x Upsampled Filtered Baseband Signal')
plt.show()

f, Px = signal.welch(sigUp, Fs, return_onesided=False, detrend=False, scaling='spectrum')
plt.semilogy(f,Px)
plt.title('PSD of Modulated 8x Upsampled Filtered Upconverted 2.5GHz Signal')
plt.show()

# Appending the LTS to the start of the signal after Upsampling and Upconversion
sig = np.append(LTS, sigUp)
nsamps = len(sig)

# =============================================================================
# UNCOMMENT THIS TO SEE THE DOWNCONVRETED PSD BEFORE TRANSMISSION
# sigDown = functions.downconversionv2(sigUp, Fc, Fs, sps, nSymbs)
# f,Px = signal.welch(sigDown, Fs, return_onesided=False, detrend=False, scaling='spectrum')
# plt.semilogy(f,Px)
# plt.title('PSD of Modulated 8x Upsampled Filtered Up/Transmitted/Down Converted Baseband Signal')
# plt.show()
# =============================================================================


# =============================================================================
# UNCOMMENT THIS TO ACTUALLY SEND THE DATA
# delay = 10e6
# txStream= sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
# rxStream= sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
# ts= int(sdr.getHardwareTime() + delay) #give us delay ns to set everything up.
# txFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
# sdr.activateStream(txStream)
# sr= sdr.writeStream(txStream, [sig.astype(np.complex64)], len(sig), txFlags, timeNs=ts)
# print(sr.ret)
# if sr.ret!= len(sig):
#     print("Bad Write!!!")
#     
# #ts= int(sdr.getHardwareTime() + delay)
# sampsRecv= np.empty(nsamps, dtype=np.complex64)
# rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
# sdr.activateStream(rxStream, rxFlags, ts, nsamps)
# sr= sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(1e6))
# print(sr.ret)
# if sr.ret!= nsamps:
#     print("Bad read!!!")
# =============================================================================
sampsRecv = sig    
plt.plot(sampsRecv[0:lenLTS])#:5000]) #plots the first 50 (nongarbage samples)
plt.title('Rx')
plt.show()

# Downconvert
sigDown = functions.downconversionv2(sampsRecv[lenLTS:], Fc, Fs, sps, nSymbs)
f,Px = signal.welch(sigDown, Fs, return_onesided=False, detrend=False, scaling='spectrum')
plt.semilogy(f,Px)
plt.title('PSD of Modulated 8x Upsampled Filtered Up/Transmitted/Down Converted Baseband Signal')
plt.show()

# LTS Stuff
cc = np.correlate(LTS_FFT,sig[0:lenLTS],"full")
plt.plot(abs(cc))             
plt.title('Cross Correlation LTS')
plt.xlabel('Lag')
plt.ylabel('Cross Correlation')
plt.show()

# Downsample by a factor of 8
sigDecDown = functions.decimate(sigDown, upSampleFactor)

plt.stem(sig[0:40])
hold()
plt.stem(sigDecDown[0:40])#[70:5000]) #plots the first 50 (nongarbage samples)
plt.title('Downconverted, Downsampled Signal vs Oringinal Data Symbols')
plt.show()

# plotting the constellation diagram
iSamps = np.real(sigDecDown)#[70:3000])#[70:])
qSamps = np.imag(sigDecDown)#[70:3000])#[70:])
plt.scatter(iSamps,qSamps)
plt.title('Received Signal Constellation')
plt.show()












