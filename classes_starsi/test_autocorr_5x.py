#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:52:25 2020

@author: simona
"""
# from stack overflow

import numpy
import matplotlib.pyplot as plt

def autocorr1(x,lags):
    '''numpy.corrcoef, partial'''

    corr=[1. if l==0 else numpy.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return numpy.array(corr)

def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=numpy.mean(x)
    var=numpy.var(x)
    xp=x-mean
    corr=[1. if l==0 else numpy.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]

    return numpy.array(corr)

def autocorr3(x,lags):
    '''fft, pad 0s, non partial'''

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**numpy.ceil(numpy.log2(ext_size)).astype('int')

    xp=x-numpy.mean(x)
    var=numpy.var(x)

    # do fft and ifft
    cf=numpy.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=numpy.fft.ifft(sf).real
    corr=corr/var/n

    return corr[:len(lags)]

def autocorr4(x,lags):
    '''fft, don't pad 0s, non partial'''
    mean=x.mean()
    var=numpy.var(x)
    xp=x-mean

    cf=numpy.fft.fft(xp)
    sf=cf.conjugate()*cf
    corr=numpy.fft.ifft(sf).real/var/len(x)

    return corr[:len(lags)]

def autocorr5(x,lags):
    '''numpy.correlate, non partial'''
    mean=x.mean()
    var=numpy.var(x)
    xp=x-mean
    corr=numpy.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]

def autocorrelation (x,la) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-numpy.mean(x)
    f = numpy.fft.fft(xp)
    p = numpy.array([numpy.real(v)**2+numpy.imag(v)**2 for v in f])
    pi = numpy.fft.ifft(p)
    return numpy.real(pi)[:x.size]/numpy.sum(xp**2)

def autocorr6(x,la):
    r2=numpy.fft.ifft(numpy.abs(numpy.fft.fft(x))**2).real
    c=(r2/x.shape-numpy.mean(x)**2)/numpy.std(x)**2
    return c[:len(x)]

if __name__=='__main__':

#    y=[28,28,26,19,16,24,26,24,24,29,29,27,31,26,38,23,13,14,28,19,19,\
#            17,22,2,4,5,7,8,14,14,23]
#    y=numpy.array(y).astype('float')

    tmp = numpy.arange(10)
    y = numpy.zeros(0)
    for i in range(5):
        y = numpy.append(y,tmp)
    
    lags=range(49)
    fig,ax=plt.subplots()

#    for funcii, labelii in zip([autocorr1, autocorr2, autocorr3, autocorr4,
#        autocorr5], ['np.corrcoef, partial', 'manual, non-partial',
#            'fft, pad 0s, non-partial', 'fft, no padding, non-partial',
#            'np.correlate, non-partial']):
    
    for funcii, labelii in zip([autocorr1, autocorr2, autocorr5, autocorr6], ['np.corrcoef, partial', 'manual, non-partial',
            'np.correlate, non-partial','fft']):

        cii=funcii(y,lags)
        print(labelii)
        print(cii)
        ax.plot(cii,label=labelii)

    ax.set_xlabel('lag')
    ax.set_ylabel('correlation coefficient')
    ax.legend()
    plt.show()