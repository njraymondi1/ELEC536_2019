# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:28:16 2019

@author: nr29
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal 
from sympy.combinatorics.graycode import GrayCode
import random

def gen_symbols(sig, mod_type):
    # Supports PSK and QAM square constellations
    # Generate Ns random symbols for a given modulation type
    #Inputs:
    #   input bit sequence
    #   mod_type - tuple of form ('PSK', 16)
    #Output:
    #   normalized sequence of symbols.
    # PSK
    if mod_type[0].lower() == 'psk':
        M = int(math.log2(mod_type[1]))
        # create gray code for M-ary
        gray = GrayCode(M)
        gray = list(gray.generate_gray())
        i = math.floor(len(sig)/M)   # in the future zero pad signal to be of length 2^...
        j = 0
        x_ints = np.arange(i)
        while j < i:
            temp = sig[j*M:(j+1)*M]
            temp = str(temp)
            temp = temp.replace("[", "")
            temp = temp.replace("]", "")
            temp = temp.replace(" ", "")
            ints = gray.index(temp)
            x_ints[j] = ints
            j = j + 1

        symbols = np.exp(-1j*2*np.pi/mod_type[1]*x_ints)*0.5 # max amp 1/2
        # print(max(abs(symbols))) # checking 0.5 normalization
    
    # QAM
    elif mod_type[0].lower() == 'qam':
        
        # Supported QAM modulations
        supported = [4,16,32,64,256]

        # Make sure valid request
        try:
            assert mod_type[1] in supported
        except:
            print("QAM constellations supported " + str(supported))
            return

# =============================================================================
#        # Define the constellation locations by i and q ints

# =============================================================================
#         i_ints = np.random.randint(0, np.sqrt(64), sig)
#         i_ints = 2*i_ints - (np.sqrt(64)-1)
#         q_ints = np.random.randint(0, np.sqrt(64), sig)
#         q_ints = 2*q_ints - (np.sqrt(64)-1)
# =============================================================================
        N = int(np.sqrt(mod_type[1]))
        M = int(math.log2(mod_type[1]))
         # create gray code for M-ary
        gray = GrayCode(M/2)
        gray = list(gray.generate_gray())
        i = math.floor(len(sig)/M)   # number of symbols req to rep data
        j = 0
        i_ints = np.arange(i)
        q_ints = np.arange(i)
        while j < i:
            temp = sig[j*M:(j+1)*M]
            temp = str(temp)
            temp = temp.replace("[", "")
            temp = temp.replace("]", "")
            temp = temp.replace(" ", "")
            ints1 = gray.index(temp[0:int(M/2)])
            ints2 = gray.index(temp[int(M/2):int(M)])
            i_ints[j] = ints1
            q_ints[j] = ints2
            j = j + 1
           
        i_ints = 2*i_ints - (np.sqrt(mod_type[1])-1)
        q_ints = 2*q_ints - (np.sqrt(mod_type[1])-1)
# =============================================================================
        symbols = i_ints + 1j*q_ints
        
        # normalize to max amplitude of 0.5
        z = max(abs(symbols))
        symbols = symbols / (2*z)
        # print(max(abs(symbols)))  # check normalization
        

    # Invalid requests
    else:
        print("Unsupported modulation type.")

    return symbols #/ np.sqrt(avg_energy(symbols))


def avg_energy(x):
    return 1/float(len(x)) * np.dot(x, np.conjugate(x)).real

def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return math.ceil(math.log(number,2))

def cpx_awgn(x, mean, var):
    # Adds complex AWGN to a signal
    n = (np.random.normal(mean, var, len(x)) + 1j*np.random.normal(mean, var, len(x)))
    return x+n

def upsample(x, rep):
    # repeats each element in signal x rep times
    # returns a signal of length len(x)*rep
    xnew = np.repeat(x, rep)
    return xnew
    
def decimate(x, rep):
    # takes signal x and removes rep samples for each sample
    # returns a signal of length len(x)/rep
    xnew = x[:x.size:rep]
    return xnew

def upconversion(x, freq, interp=1):
    # upconverts signal x to frequency freq
    # x should be the symbol vector so we can use IQ
    # returns xnew as symbol vector where 
    # Inew[n] = I[n]cos[2pi*freq*n], Qnew[n] = -Q[n]sin[2pi*freq*n]
    # interp is the interpolation factor and is how many samples of the 
    # carrier we will use between data samples, default 1
    
    # generate carrier
    #time = np.arange(len(x))/interp
    time = np.linspace(0, 2*np.pi, len(x))
    carCos = np.cos(2*np.pi*freq*time)
    carSin = -np.sin(2*np.pi*freq*time)
    I_old = np.real(x)
    Q_old = np.imag(x)
    I = I_old*carCos
    Q = Q_old*carSin
    xnew = I + 1j*Q
    return xnew

def downconversion(x, freq, interp=1):
    # downconvert signal x from frequency freq
    # x should be the symbol vector so we can use IQ
    # returns xnew as symbol vector where 
    # Inew[n] = I[n]cos[-2pi*freq*n], Qnew[n] = -Q[n]sin[-2pi*freq*n]
    # interp is the interpolation factor and is how many samples of the 
    # carrier we will use between data samples, default 1
    
    # generate carrier
    #time = np.arange(len(x))/interp
    time = np.linspace(0, 2*np.pi, len(x))
    carCos = np.cos(-2*np.pi*freq*time)
    carSin = -np.sin(-2*np.pi*freq*time)
    I_old = np.real(x)
    Q_old = np.imag(x)
    I = I_old*carCos
    Q = Q_old*carSin
    xnew = I + 1j*Q
    return xnew

# =============================================================================
# L = 10000    # length of binary signal
# sig = np.random.randint(0, 2, L)
# symbols = gen_symbols(sig,('psk',16))
# plt.scatter(np.real(symbols),np.imag(symbols))
# plt.title('No Noise')
# plt.show()
# 
# # add sero mean, var=0.05 gaussian noise to the signal 
# noiseSig = cpx_awgn(symbols, 0, 0.05)
# plt.scatter(np.real(noiseSig),np.imag(noiseSig))
# plt.title('Noisy - Zero Mean 0.05 Var Gaussian')
# plt.show()
# =============================================================================

# =============================================================================
# L = 10   # length of binary signal
# sig = np.random.randint(0, 3, L)
# plt.stem(sig)
# plt.show()
# sigN  = upconversion(sig, 500e9)
# #plt.plot(sigN)
# #plt.show()
# sigN = signal.resample(sigN, len(sigN)*10)
# plt.plot(sigN)
# plt.title('resampled upsampled signal')
# plt.show()
# sigNN = downconversion(sigN, 500e9)
# sigNN = signal.resample(sigNN, len(sig))
# plt.stem(sigNN)
# plt.show()
# =============================================================================












