# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:21:00 2019

@author: nr29
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal 
from sympy.combinatorics.graycode import GrayCode
import random
import functions

def modulate(inputSymbols):
    # OFDM modulation for input signal x
    K = 64 #subcarriers
    subs = np.arange(K)
    #zeroSubs = np.array([0,1,2,3,4,5,32,60,61,62,63,64])
    subs[1] = 0
    subs[2] = 0
    subs[3] = 0
    subs[4] = 0
    subs[5] = 0
    subs[32] = 0
    subs[60] = 0
    subs[61] = 0
    subs[62] = 0
    subs[63] = 0
    subs[59] = 0
    #print(subs)
    dataSubs = subs[subs != 0]
    # inputSymbols is the symbols from another modulation scheme (dqpsk for instance) from another function
    OFDMsymbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    #OFDMsymbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    OFDMsymbol[dataSubs] = inputSymbols  # allocate the pilot subcarriers
    return OFDMsymbol

# =============================================================================
# L = 1000                          # length of binary signal
# sig = np.random.randint(0, 2, L)        # generate random msg bits
# binarySig = sig                 # save this for later
# #symbols = functions.gen_symbols(sig,('qam',4))
# symbols = functions.dqpskMod(sig)
# symbols = symbols[0:64-12]
# print(symbols[5:10])
# OFDMsymbols = modulate(symbols)
# print(OFDMsymbols[5:10])
# # perform ifft of OFDM data
# 
# iSamps = np.real(OFDMsymbols)#[70:3000])#[70:])
# qSamps = np.imag(OFDMsymbols)#[70:3000])#[70:])
# plt.scatter(iSamps,qSamps)
# plt.title('Received Signal Constellation')
# plt.show()
# =============================================================================
