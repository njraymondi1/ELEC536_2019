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
from itertools import chain

def gen_symbols(sig, mod_type):
    # Supports PSK, APSK (16,32), QAM (4,16,64,256) (square constellations)
    # Generate Ns random symbols for a given modulation type
    #Inputs:
    #   input bit sequence
    #   mod_type - tuple of form ('APSK', 16)
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
    # xnew = np.repeat(x, rep)
    ins = np.repeat(0, rep-1)
    xnew = np.insert(x, 0, ins)
    return xnew

def upsampleZeros(x, rep):
    # upsamples a signal by a factor of rep by adding
    # rep-1 zeros between each value
    # YEA I FUDGED IT TO HANDLE UPSAMPLING FACTOR OF 8, SUE ME
    z = np.insert(x, slice(1, None), 0)
    z = np.insert(z, slice(1, None), 0) 
    z = np.insert(z, slice(1, None), 0) 
    z = np.append(z, np.zeros(7))
    return z
    
def decimate(x, rep):
    # takes signal x and removes rep samples for each sample
    # returns a signal of length len(x)/rep
    #print(x.dtype)
    xnew = x[:x.size:rep]
    #print(xnew.dtype)
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

def upconversionv2(x, Fc, Fs, sps, frameLength):
    Rsym = Fs/sps
    t = np.arange(0,(frameLength/Rsym)-1/(2*Fs),1/Fs)
    carrier = np.sqrt(2)*np.exp(1j*2*np.pi*Fc*t)
    xUp = np.multiply(x, carrier)
    xUp = np.real(xUp)
    return xUp

def downconversionv2(x, Fc, Fs, sps, frameLength):
    Rsym = Fs/sps
    t = np.arange(0,(frameLength/Rsym)-1/(2*Fs),1/Fs)
    carrier = np.sqrt(2)*np.exp(1j*2*np.pi*Fc*t)
    carrier = np.conj(carrier)
    xDown = np.multiply(x, carrier)
    #print(xDown.dtype)
    return xDown

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

def rrcosFilter(length, alpha):
    # output coefficients for rrcf of length with certain alpha
    # TO DO
    #      only accepts length = 32
    #      only accepnts alpha = [0.25, 0.5, 0.75]

    if alpha == 0.25:
        coeffs = np.array([0.0188,0.0018,-0.0195,-0.0417,-0.0605,-0.0715,-0.0706,-0.0548,-0.0228,0.0246,
                           0.0845,0.1520,0.2209,0.2842,0.3351,0.3681,0.3796,0.3681,0.3351,0.2842,0.2209,
                           0.1520,0.0845,0.0246,-0.0228,-0.0548,-0.0706,-0.0715,-0.0605,-0.0417,-0.0195,0.0018,0.0188])
    elif alpha == 0.5:
        coeffs = np.array([-0.0005,0.0000,0.0007,-0.0000,-0.0010,0.0001,0.0014,-0.0001,-0.0022,0.0002,
                         0.0039,-0.0006,-0.0088,0.0026,0.0370,-0.0924,0.9899,-0.0924,0.0370,0.0026,
                         -0.0088,-0.0006,0.0039,0.0002,-0.0022,-0.0001,0.0014,0.0001,-0.0010,-0.0000,
                         0.0007,0.0000,-0.0005])
    elif alpha == 0.75:
        coeffs = np.array([-0.0003,-0.0003,0.0000,0.0004,0.0006,0.0005,-0.0000,-0.0008,-0.0014,-0.0012,
                          0.0001,0.0026,0.0055,0.0062,-0.0037,-0.0926,0.9913,-0.0926,-0.0037,0.0062,
                          0.0055,0.0026,0.0001,-0.0012,-0.0014,-0.0008,-0.0000,0.0005,0.0006,0.0004,
                          0.0000,-0.0003,-0.0003])
    return coeffs

def BERcalc(sig1, sig2):
    # compares binary signals sig1 and sig2 and calculates the BER
    # i.e. the total # of errors / the length of the signals
    
    # sig1 and sig2 should have the same length
    L = len(sig1)
    xor = np.bitwise_xor(sig1,sig2)
    numErrors = np.sum(xor)
    BER = numErrors/L
    return BER

def cumBER(sig1, sig2):
    # finds cumulative ber signal
    # i.e. for a bit error string [0 0 1 0 1]
    # returns                     [0 0 1 1 2]
    xor = np.bitwise_xor(sig1,sig2)
    cum = [sum(xor[:i+1]) for i in range(len(xor))]
    return cum

def dqpskMod(sig):
    # performs differential qpsk mod on input sig
    
    # table of phase difference references
    A = '00'#0               # corresponds to 00 
    B = '10'#-np.pi/2        # corresponds to 10 
    C = '01'#np.pi/2         # corresponds to 01 
    D = '11'#np.pi           # corresponds to 11 
    
    M = int(2)
    # create gray code for M-ary
    gray = GrayCode(M/2)
    gray = list(gray.generate_gray())
    i = math.floor(len(sig)/M)   # number of symbols req to rep data
    diff = 0
    
    # reference symbol A to start
    j = 0
    symbols = np.arange(i+1,dtype=complex)
    symbols[0] = 1
    while j < i:
        temp = sig[j*M:(j+1)*M]
        temp = str(temp)
        temp = temp.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace(" ", "")
        if temp == A:
            diff = 0
        elif temp == B:
            diff = -np.pi/2
        elif temp == C:
            diff = np.pi/2
        else:
            diff = np.pi
        symbols[j+1] = symbols[j]*np.exp(-1j*2*(np.pi - diff/2))
        #print(symbols[j])
        j = j + 1
        
    return symbols[1:]    

def takeClosest(num,collection):
    return min(collection,key=lambda x:abs(x-num))

def dqpskDeMod(symbs):
    # performs soft demod of input symbol vector to get back original bit sequence
    
    # Lookup table of phases
    A = 0               # corresponds to 00 
    B = -np.pi/2        # corresponds to 10 
    C = np.pi/2         # corresponds to 01 
    D = np.pi           # corresponds to 11 
    E = -(np.pi + np.pi/2) # eat me I suck at coding and don't deny it
    # we know the first symbol will be A or '00' 
    
    angles = np.angle(symbs)
    for i in range(len(angles)):
        if takeClosest(np.negative(angles[i]),[A,B,C,D]) == D:
            angles[i] = np.pi
            
    angleDiff = np.negative([x-y for x, y in zip(angles, angles[1:])])
    angleDiff = np.insert(angleDiff, 0, angles[0])
    for i in range(len(angleDiff)):
        if takeClosest(np.negative(angleDiff[i]),[A,B,C,D,E]) == E:
            angleDiff[i] = -np.pi/2
        elif takeClosest(np.negative(angleDiff[i]),[A,B,C,D]) == D:
            angleDiff[i] = np.pi
            
    Abit = np.array([0,0])
    Bbit = np.array([1,0])
    Cbit = np.array([0,1])
    Dbit = np.array([1,1])
    constPoints = np.arange(len(angles),dtype=complex)
    signal = []
    # pick closest constellation point
    for i in range(len(angles)):
        constPoints[i] = takeClosest(angleDiff[i],[A,B,C,D])
        if constPoints[i] == A:
            signal = np.append(signal,Abit)
        elif constPoints[i] == B:
            signal = np.append(signal,Bbit)
        elif constPoints[i] == C:
            signal = np.append(signal,Cbit)
        else:
            signal = np.append(signal,Dbit) 
        signal = np.ndarray.astype(signal, int)
    return signal
    
def secondPeakLTS(cc):
    # finds the second peak of the LTS correlation when fed the abs(cc) function
    indices = sorted( [(x,i) for (i,x) in enumerate(cc)], reverse=True )[:2] 
    indices = np.array(indices)
    indices = (indices[0,1], indices[1,1])
    secondPeak = max(indices)
    secondPeak = int(secondPeak)
    return secondPeak

def removeDC(sig):
    iSamps = np.real(sig)
    qSamps = np.imag(sig)
    DC = np.mean( np.array([ iSamps, qSamps ]) )
    sigDC = sig - DC
    return sigDC

def alamouti(symbols):
    # returns the corresponding second branch of the alamouti code
    # for input symbols {s_i}, i = 1,...,M
    # returns symbols {-s2*, s1*, -s4*, s3*...}
    symb = np.reshape(symbols,(-1,2))
    arr = np.array(np.zeros((len(symb),2)),dtype=np.complex128)
    for i in range(len(symb)):
        s2 = np.conj(symb[i][0])
        s1 = -np.conj(symb[i][1])
        arr[i] = np.array([s1,s2],dtype=np.complex128)
    alamoutiSeq = np.reshape(arr,(1,len(symbols)))
    return alamoutiSeq

def alamoutiv2(symbols):
    # takes a symbol vector Symbols and returns 2 alamouti 2x1 coded vectors where
    # symbs = [s1  s2  s3 s4 ...]
    # vec1 = [s1 -s2* s3 -s4*...]
    # vec2 = [s2 s1*  s4 s3* ...]
    freq = np.fft.fft(symbols)
    symbs = np.reshape(symbols,(-1,2))
    freq = np.reshape(freq,(-1,2))
    arr1 = np.array(np.zeros((len(symbs),2)),dtype=np.complex128)
    arr2 = np.array(np.zeros((len(symbs),2)),dtype=np.complex128)
    for i in range(len(symbs)):
        # here notation means symbol{antenna}{time index}
        #arr1[i][0] = freq[i][0]
        #arr1[i][1] = -np.conj(freq[i][1]) ###############################################
        #arr2[i][0] = freq[i][1]
        #arr2[i][1] = np.conj(freq[i][0])
        arr1[i][0] = symbs[i][0]
        arr1[i][1] = np.conj(-symbs[i][1]) ###############################################
        arr2[i][0] = symbs[i][1]
        arr2[i][1] = np.conj(symbs[i][0])
    vec1 = np.reshape(arr1,(1,len(symbols)))
    vec2 = np.reshape(arr2,(1,len(symbols)))
    vec1 = list(chain(*vec1))
    vec2 = list(chain(*vec2))
    #vec1 = np.fft.ifft(vec1)
    #vec2 = np.fft.ifft(vec2)
    return vec1, vec2

#-------------------------4QAM-------------------------------------------------
def QAMmod4(bits):
    mapping_table = {
    (0,0) : -0.35355339-0.35355339j,
    (0,1) : -0.35355339+0.35355339j,
    (1,0) :  0.35355339-0.35355339j,
    (1,1) :  0.35355339+0.35355339j}
    bits = bits.reshape((int(len(bits)/2), 2))
    return np.array([mapping_table[tuple(b)] for b in bits])

def QAMdemod4(QAM):
    mapping_table = {
    (0,0) : -0.35355339-0.35355339j,
    (0,1) : -0.35355339+0.35355339j,
    (1,0) :  0.35355339-0.35355339j,
    (1,1) :  0.35355339+0.35355339j}
    demapping_table = {v : k for k, v in mapping_table.items()}
    # this code courtesty of Python Illustrations' OFDM page - adapted for 4QAM
    # https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html
    constellation = np.array([x for x in demapping_table.keys()])
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    const_index = dists.argmin(axis=1)
    hardDecision = constellation[const_index]
    PS_est = np.vstack([demapping_table[C] for C in hardDecision])
    return np.array(PS_est.reshape((-1,))), hardDecision
#------------------------------------------------------------------------------



# =============================================================================
# L = 16                         # length of binary signal
# sig = np.random.randint(0, 2, L) 
# print(sig)
# symbols = QAMmod4(sig)
# print(symbols)
# bits, symbDecisions = QAMdemod4(symbols)
# print(bits)
# =============================================================================

# =============================================================================
# L = 8                         # length of binary signal
# sig = np.random.randint(0, 2, L) 
# symbols = QAMmod4(sig)
# print(symbols)
# print(len(symbols))
# ala1, ala2 = alamoutiv2(symbols)
# print(ala1)
# print(len(ala1))
# print(ala2)
# print(len(ala2))
# =============================================================================









