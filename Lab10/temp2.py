# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:45:18 2019

@author: nr29
"""

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
import OFDM

sdr= SoapySDR.Device(dict(driver="iris", serial = "RF3C000044"))
chan = 0
chan2 = 1

sdr.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdr.setSampleRate(SOAPY_SDR_RX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_RX, chan, 50)

sdr.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdr.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan, 35)

sdr.setAntenna(SOAPY_SDR_TX, chan2, "TRX")
sdr.setSampleRate(SOAPY_SDR_TX, chan2, 10e6)
sdr.setFrequency(SOAPY_SDR_TX, chan2, "RF", 2.45e9)
sdr.setGain(SOAPY_SDR_TX, chan2, 35)


# =============================================================================
# # generating the signal 4qam from mod_gen function file
# L = 1000                                # length of binary signal
# sig = np.random.randint(0, 2, L)        # generate random msg bits
# binarySig = sig                         # save this for later
# print(binarySig[0:20])
# Original_symbols = functions.QAMmod4(binarySig)
# sig = Original_symbols
# nSymbs = len(sig)
# nsamps = len(sig) # this is now nSymbs
# =============================================================================

#------------------------------------------------------------------------------
# import file to send
f = open('textfile.txt', 'r')
content = f.read()
f.close()

# =============================================================================
# a_bytes = bytes(content, "ascii")
# binary1 = ' '.join(["{0:b}".format(x) for x in a_bytes])
# binary2 = binary1.replace(" ","")
# binary3 = list(map(int, binary2))
# binary4 = np.asarray(binary3)
# binarySig = binary4                         # save this for later
# Original_symbols = functions.QAMmod4(binarySig)
# sig = Original_symbols
# nSymbs = len(sig)
# nsamps = len(sig) # this is now nSymbs
# =============================================================================

binary1 = functions.string2bits(content)
binary2 = ''.join(str(e) for e in binary1)
binary3 = list(map(int, binary2))
binary4 = np.asarray(binary3)
binarySig = binary4                         # save this for later
Original_symbols = functions.QAMmod4(binarySig)
Reference_symbols = Original_symbols
if len(Original_symbols) % 52 != 0:
    Original_symbols = np.append(Original_symbols, np.zeros(52-(len(Original_symbols) % 52)))
numOFDMframes = int(len(Original_symbols)/52)

sig_data = np.zeros(numOFDMframes*64, dtype=complex)
for i in range(numOFDMframes):
    sig_data[i*64:(i+1)*64] = OFDM.modulate(Original_symbols[52*i:52*(i+1)])
sig = np.fft.ifft(sig_data)*5  #OFDM modulation, ifft, give it a little extra gain - THIS IS TIME DOMAIN
nSymbs = len(sig)
nsamps = len(sig) # this is now nSymbs

Fc = 2.5e9               
span = 4
Fs = 20e9
sps = 8
Rsym = Fs/sps

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

sampsRecv = sig*(1+1j)                 #uncomment this to remove transmission channel
plt.plot(sampsRecv[0:500])
plt.title('Recevied Signal 1 - No operations')
plt.show()

#------------------------------------------------------------------------------
# LTS Stuff
#cc = np.correlate(sampsRecv[0:int(len(sampsRecv/2)),0],LTS_single,'full')
cc = np.correlate(sampsRecv,LTS_single,'full')   
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

# OFDM demodulation
#OFDMdemod = np.fft.fft(sampsRecv[endLTS:endLTS+nSymbs*sps])

#------------------------------------------------------------------------------


# Downsample our equalized signal by a factor of 8-----------------------------
#sigDecDown = signal.decimate(sampsRecv[endLTS:endLTS+nSymbs*sps], upSampleFactor, ftype='fir')
#sigDecDown = signal.decimate(OFDMdemod, upSampleFactor, ftype='fir')
sigDecDown = signal.decimate(sampsRecv[endLTS:endLTS+nSymbs*sps], upSampleFactor, ftype='fir')
sigDecDown = np.fft.fft(sigDecDown)
plt.scatter(np.real(sigDecDown),np.imag(sigDecDown))
plt.title('Received Constellation Signal 1')
plt.show()
plt.scatter(np.real(Original_symbols),np.imag(Original_symbols))
plt.title('Original Symbol Constellation')
plt.show()
#------------------------------------------------------------------------------
#OFDMdemod = np.fft.fft(sigDecDown)


# equalize by dividing received signal by channel estimate
eqSig1 = np.zeros(len(sigDecDown), dtype=complex)
for i in range(numOFDMframes):
    eqSig1[i*64:(i+1)*64] = sigDecDown[i*64:(i+1)*64]*(chanEstimate/3.25)
    
# recevier
sumSig = eqSig1
sigdeMod = np.zeros(len(Original_symbols), dtype=complex)
for i in range(52): #number of sC
    sigdeMod[i*52:(i+1)*52] = OFDM.OFDMdemodulate(sumSig[i*64:(i+1)*64])


plt.scatter(np.real(sigdeMod),np.imag(sigdeMod))
plt.title('Equalized Recovered Constellation')
plt.show()

asdf

# demodulation
[sig2bits,sig2symbs] = functions.QAMdemod4(sumSig)
BERsig = functions.BERcalc(binarySig, sig2bits)
print('BER')
print(BERsig)

asdf
# convert our recevied signal back into the original file
# i.e. cnvert sig2bits into a .txt file

j = 0
i = int(len(sig2bits)/8)
strings = list()
while j<i:
    temp = sig2bits[j*8:(j+1)*8]
    temp = str(temp)
    temp = temp.replace("[", "")
    temp = temp.replace("]", "")
    temp = temp.replace(" ", "")
    strings.append(temp)
    j = j + 1
      
z = functions.bits2string(strings)
outputString = str(z)

text_file = open("Output.txt", "w")
text_file.write(outputString)
text_file.close()