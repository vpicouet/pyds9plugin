#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:54:58 2019

@author: dvibert
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang
from scipy.signal import convolve, deconvolve

N = 20
#N = 10
original = np.random.poisson(2,size=N)
data = np.zeros_like(original)
nz = original>0
data[nz] = erlang.rvs(original[nz],scale=1000.)

image = np.random.poisson(2,size=(2069,3216))
nz = image>0
image[nz] = erlang.rvs(image[nz],scale=1000.)


smearing_length = 1.5 #pix 
smearing_p = np.exp(-1/smearing_length) # deferred charge proba
smearing = (1-smearing_p)/(1-np.power(smearing_p,6))* np.power(smearing_p,np.arange(6))
recorded = convolve(smearing, data)
recovered, remainder = deconvolve(recorded, smearing)

plt.figure()
plt.plot(data, label= 'data')
plt.plot(recorded, label= 'smeared data')
plt.plot(recovered, label='deconvolved')
plt.legend()

plt.figure()
plt.plot(recovered - data)
print((recovered - data).std())

noise = np.random.normal(0,scale=100,size=N+5)
recorded_noise = recorded + noise
recovered_noise, remainder_noise = deconvolve(recorded_noise, smearing)

plt.figure()
plt.plot(data, label= 'data')
plt.plot(recorded_noise, label= 'smeared data + noise')
plt.plot(recovered_noise, label='deconvolved')
plt.legend()

plt.figure()
plt.plot(recovered_noise - data)
print((recovered_noise - data).std())

## convolution with varrying kernel
from scipy.sparse import dia_matrix
#offsets = np.arange(0,-6,-1)
offsets = np.arange(6)
#diags = smearing[:,np.newaxis].repeat(N+5,axis=1)
#A = dia_matrix((diags, offsets), shape=(N+5,N+5))

#variing smearing length
# l = l0 * exp(- F/a) ; a=50000 l0=1.54 (vol)
def set_smearing_kernels(data, zero_length, smearing_p, mean):
    # level of photoelectrons above which no semaring
    smearing_p_var = np.where(data > zero_length, 0., smearing_p*((zero_length - data)/(zero_length - mean)))
    smearing_kernels = (1-smearing_p_var)/(1-np.power(smearing_p_var,6))* np.power(smearing_p_var,np.arange(6)[:,np.newaxis])
    return smearing_kernels

def VariableSmearingKernels(data, Smearing=1.5, SmearExpDecrement=50000):

    smearing_length = Smearing*np.exp(-data/SmearExpDecrement)
    smearing_kernels = np.exp(-np.arange(6)[:,np.newaxis]/smearing_length )
    smearing_kernels /= smearing_kernels.sum(axis=0)

    return smearing_kernels

#zero_length = 10000
#mean = 2000
#smearing_kernels = set_smearing_kernels(data, zero_length, smearing_p, mean)

smearing_kernels = VariableSmearingKernels(data, Smearing=1.5, SmearExpDecrement=50000)

A = dia_matrix((smearing_kernels, offsets), shape=(N,N))


#smeared = A.dot(np.concatenate((data,np.zeros(5))))
smeared = A.dot(data)
smeared_noise = smeared + noise[:N]

plt.figure()
#plt.plot(test - recorded)
plt.plot(data)
plt.plot(recorded)
plt.plot(smeared)
plt.legend(labels=[ 'data', 'fixed smearing',  'var smearing'   ]) 

# deconvolve
#import scipy.linalg as LA
#from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg import solve_banded

#test = LA.solve_triangular(A, smeared, lower=True )
#test_noise = LA.solve_triangular(A, recorded_noise, lower=True )
#test = spsolve_triangular(A, smeared, lower=True)
#test = solve_banded((5,0), smearing_kernels, smeared)
#test_noise = solve_banded((5,0), smearing_kernels, smeared_noise)

# invert knowing the kernel
test = solve_banded((0,5), smearing_kernels[::-1,:], smeared)
test_noise = solve_banded((0,5), smearing_kernels[::-1,:], smeared_noise)

print(np.sqrt(np.square(test[:N] - data).mean()))
print(np.sqrt(np.square(test_noise[:N] - data).mean()))

plt.figure()
plt.plot(data)
plt.plot(smeared_noise)
plt.plot(test)
plt.plot(test_noise)
plt.legend(labels=[  'data', 'smeared + noise', 'unsmeared (wo noise)', 'unsmeared (w noise)'   ]) 


# interative invert wo knowing the kernel
##############################################
print(np.sqrt(np.square(smeared - data).mean()))
test = smeared
for i in range(10): 
    smearing_kernels = set_smearing_kernels(test, zero_length, smearing_p, mean)
    test = solve_banded((0,5), smearing_kernels[::-1,:], smeared)
    print(np.sqrt(np.square(test[:N] - data).mean()))

plt.figure()
plt.plot(data)
plt.plot(smeared)
plt.plot(test)
plt.legend(labels=[  'data', 'smeard', 'unsmeared'  ]) 

# with noise
print(np.sqrt(np.square(smeared_noise - data).mean()))
test = smeared_noise
for i in range(5): 
    smearing_kernels = set_smearing_kernels(test, zero_length, smearing_p, mean)
    test = solve_banded((0,5), smearing_kernels[::-1,:], smeared_noise)
    print(np.sqrt(np.square(test[:N] - data).mean()))

plt.figure()
plt.plot(data)
plt.plot(smeared_noise)
plt.plot(test)
plt.legend(labels=[  'data', 'smeared + noise', 'unsmeared'  ]) 








plt.figure()
plt.plot(test[:N] - recovered)
plt.plot(test_noise[:N] - recovered_noise)


# deconvolve with tikhonov regularization
#Anormal = A.T.dot(A)
band_square = convolve(smearing,smearing[::-1])[5:]

diags_square = band_square[:,np.newaxis].repeat(N+5,axis=1)
damp = 0#1e-16 # sigma²
diags_square[0] += damp
#LA.eig_banded(diags_square, lower=True, eigvals_only=True)

cho = LA.cholesky_banded(diags_square, lower=True)

rhs = A.T.dot(recorded) #/damp
test2 = LA.cho_solve_banded((cho, True), rhs)
 
rhs_noise = A.T.dot(recorded_noise)#/damp
test2_noise = LA.cho_solve_banded((cho, True), rhs_noise)
print((test2_noise[:N] - data).std())

plt.figure()
plt.plot(test2[:N] - recovered)

plt.figure()
plt.plot(data)
#plt.plot(recorded)
plt.plot(test2[:N])
plt.plot(test2_noise[:N])

plt.figure()
plt.plot(test2_noise[:N] - recovered_noise)

plt.figure()
plt.plot(test2 - test)
