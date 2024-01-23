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
from scipy.linalg import solve_banded

from matplotlib.colors import LogNorm

def deconvolve_smearing(data, smearing_length=1.4,plot_=True):
    smearing_p = np.exp(-1/smearing_length) # deferred charge proba
    smearing = (1-smearing_p)/(1-np.power(smearing_p,6))* np.power(smearing_p,np.arange(6))
    # data=data-np.nanmedian(data)
    recovered_noise, remainder_noise = deconvolve(data, smearing)
    if plot_:
        plt.figure()
        plt.title("Smearing length = %0.2f"%(smearing_length))
        plt.plot(data.T, label= 'data')
        # plt.plot(recorded_noise, label= 'smeared data + noise')
        plt.plot(np.array(recovered_noise).T, label='deconvolved')
        plt.legend()

    return recovered_noise

    a = np.loadtxt("/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/Science/smear.txt")
    # plt.imshow(np.log10(a))
    
    for i in np.arange(10):
        deconvolve_smearing(data=a[i,:50],smearing_length=1.4)

# d=DS9n()
# ds9 = d.get_arr2np()
min_value = np.nanmin(ds9)
ds9 = ds9-min_value+1
copy =ds9#[:,::-1]
lx,ly = ds9.shape
ds9 = np.ones((lx,ly-5))
for i, line in enumerate(copy):
    # print(ds9[i])
    # print(line)
    ds9[i] = deconvolve_smearing(copy[i], smearing_length=0.8,plot_=True)


#%%

# Set vmin and vmax
vmin =  np.nanmin(copy)
vmax = np.nanmax(copy)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True, sharey=True)

# Plot data in log scale
im1 = axs[0].imshow(copy, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
im2 = axs[1].imshow(ds9, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
im3 = axs[2].imshow(ds9-copy[:,:-5], cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))

# Add color bar to the right
cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im3, cax=cbar_ax)

# Set titles and labels
axs[0].set_title('Data')
axs[1].set_title('Deconvolve')
axs[2].set_title('Residual')
fig.tight_layout()
plt.show()



#%%







if 1==0:

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
    plt.plot(recovered, "--",label='deconvolved')
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



if 1==0:
#%% Vincent
    
    def create_array():
        return
    
    def deconvolve_smearing(data, smearing_length=1.4,plot_=True):
        smearing_p = np.exp(-1/smearing_length) # deferred charge proba
        smearing = (1-smearing_p)/(1-np.power(smearing_p,6))* np.power(smearing_p,np.arange(6))
        data-=np.nanmedian(data)
        recovered_noise, remainder_noise = deconvolve(data, smearing)
        if plot_:
            plt.figure()
            plt.title("Smearing length = %0.2f"%(smearing_length))
            plt.plot(data.T, label= 'data')
            # plt.plot(recorded_noise, label= 'smeared data + noise')
            plt.plot(np.array(recovered_noise).T, label='deconvolved')
            plt.legend()
    
        return recovered_noise
    
    a = np.loadtxt("/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/Science/smear.txt")
    # plt.imshow(np.log10(a))
    
    for i in np.arange(10):
        deconvolve_smearing(data=a[i,:50],smearing_length=1.4)
    
    
    #%%
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
