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
import time

sdrTx= SoapySDR.Device(dict(driver="iris", serial = "RF3C000063"))
sdrRx= SoapySDR.Device(dict(driver="iris", serial = "RF3C000031"))
chan = 0
sdrTx.setAntenna(SOAPY_SDR_TX, chan, "TRX")
sdrTx.setSampleRate(SOAPY_SDR_TX, chan, 10e6)
sdrTx.setFrequency(SOAPY_SDR_TX, chan, "RF", 2.45e9)
sdrTx.setGain(SOAPY_SDR_TX, chan, 30)
sdrRx.setAntenna(SOAPY_SDR_RX, chan, "RX")
sdrRx.setSampleRate(SOAPY_SDR_RX, chan, 40e6)
sdrRx.setFrequency(SOAPY_SDR_RX, chan, "RF", 2.45e9)
sdrRx.setGain(SOAPY_SDR_RX, chan, 35)

# =============================================================================
# def cfloat2uint32(arr, order='IQ'):
# 		arr_i = (np.real(arr) * 32767).astype(np.uint16)
# 		arr_q = (np.imag(arr) * 32767).astype(np.uint16)
# 		if order == 'IQ':
# 			return np.bitwise_or(arr_q ,np.left_shift(arr_i.astype(np.uint32), 16))
# 		else:
# 			return np.bitwise_or(arr_i ,np.left_shift(arr_q.astype(np.uint32), 16))
# =============================================================================

# generating the signal 4qam from mod_gen function file
L = 8000                               # length of binary signal
sig = np.random.randint(0, 2, L)        # generate random msg bits
symbols = lab3_functions.gen_symbols(sig,('qam',4))
#symbols = part2_b.cpx_awgn(symbols, 0, 0.05)
plt.scatter(np.real(symbols),np.imag(symbols))
plt.title('4QAM Transmitted Signal Constellation')
plt.show()
#sig = np.fft.fft(symbols)
sig = symbols
sig = lab3_functions.upsample(sig, 8) # upsample by a factor of 8
sig = lab3_functions.upconversion(sig, 2.5e6) # upconvert to 2.5MHz
nsamps = len(sig)

delay = 10e6
txStream= sdrTx.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
rxStream= sdrRx.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})


# =============================================================================
# replay_addr = 0
# max_replay = 4096  #todo: read from hardware
# if(len(sig) > max_replay):
# 	print("Warning: Continuous mode signal must be less than %d samples. Using first %d samples." % (max_replay, max_replay) )
# 	sig = sig[:max_replay]
# sdrTx.writeRegisters('TX_RAM_A', replay_addr, cfloat2uint32(sig).tolist())
# sdrTx.writeSetting("TX_REPLAY", str(len(sig)))
# =============================================================================


ts= int(sdrTx.getHardwareTime() + delay) #give us delay ns to set everything up.
txFlags= SOAPY_SDR_HAS_TIME #| SOAPY_SDR_END_BURST
sdrTx.activateStream(txStream)
sr= sdrTx.writeStream(txStream, [sig.astype(np.complex64)], len(sig), txFlags, timeNs=ts)
print(sr.ret)
if sr.ret!= len(sig):
    print("Bad Write!!!")
   
ts= int(sdrRx.getHardwareTime() + delay)
sampsRecv= np.empty(nsamps, dtype=np.complex64)
rxFlags= SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
sdrRx.activateStream(rxStream, rxFlags, ts, nsamps)
sr= sdrRx.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(1e6))
print(sr.ret)
if sr.ret!= nsamps:
    print("Bad read!!!")
# =============================================================================
# =============================================================================
# rx_delay=57
# nsamps = int(nsamps)
# txserial = 'RF3C000063'
# trig_sdr = sdrTx if txserial is not None else sdrRx
# rx_delay_ns = SoapySDR.ticksToTimeNs(rx_delay,40e6)
# hw_time = trig_sdr.getHardwareTime()
# ts = hw_time + delay + rx_delay_ns if ts is None else ts + rx_delay_ns
# sampsRecv = np.empty(nsamps, dtype=np.complex64)
# rxFlags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
# sdrRx.activateStream(rxStream, rxFlags, ts, nsamps)
# 
# #print(hw_time,delay,rx_delay_ns,ts)
# #print([sdr.getHardwareTime() for sdr in self.sdrs])
# #time.sleep((ts - hw_time)/1e9)		
# sr = sdrRx.readStream(rxStream, [sampsRecv], timeoutUs=int(1e6), numElems=1)
# if sr.ret != nsamps:
# 	print("Bad read!!!")
# =============================================================================

    
plt.plot(sig[0:50]) #plots the first 50 (nongarbage samples)
plt.title('Tx')
plt.show()  
#sampsRecV = np.fft.ifft(sampsRecv)  
# =============================================================================
plt.plot(sampsRecv[0:50]) #plots the first 50 (nongarbage samples)
plt.title('Rx')
plt.show()
# =============================================================================

sampsRecvDec = lab3_functions.decimate(sampsRecv, 8)
sampsRecvDC = lab3_functions.downconversion(sampsRecvDec, 2.5e6)
# =============================================================================
# =============================================================================
# plt.plot(sampsRecv[0:50]) #plots the first 50 (nongarbage samples)
# plt.title('Decimated Rx')
# plt.show()
# =============================================================================
# =============================================================================

# plotting the constellation diagram
iSamps = np.real(sampsRecv)#[70:])
qSamps = np.imag(sampsRecv)#[70:])
plt.scatter(iSamps,qSamps)
plt.title('Received Signal Constellation')
plt.show()

# plotting the fft
fftData = np.fft.fft(sampsRecv)
plt.plot(abs(fftData))
plt.title('FFT Recevied Signal - No Decimation or Downconversion')
plt.show()



















