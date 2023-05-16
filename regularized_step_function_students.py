#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to construct the regularized step function as required for Part 3 of the homework.

"""
import numpy as np

def reg_heavyside(x, bnd, e):
    """
    Function that construct a regularized heavyside as function of x.
    
    Parameters
    ----------
    x : array of float
        Support of the function.
    bnd : float
        Jump position.
    e : flot
        Regularization parameter.
    
    Returns
    -------
    F : array of float
        Regularized heavyside f(x).
    
    """
    F = np.zeros(len(x))
    idx = np.append(np.where((x-bnd)/e>=-1), np.where((x-bnd)/e<=1))
    F[idx] = 1/2 * (1 + (x[idx]-bnd)/e +1/np.pi * np.sin(np.pi*((x[idx]-bnd)/e)))
    F[np.where((x-bnd)/e>1)] = 1
    F[np.where((x-bnd)/e<-1)] = 0
    return F

def reg_step(x, bnd1, bnd2, e = 0.05):
    """
    Function combining two regularized heavyside to construct a regularized step function.

    Parameters
    ----------
    x : array of float
        Support of the function.
    bnd1 : float
        First jump position.
    bnd2 : float
        Second jump position.
    e : float, optional
        Regularization parameter. The default is 0.05.

    Returns
    -------
    F : array of float
        Regularized step function f(x).

    """
    A = reg_heavyside(x, bnd1, e)
    B = reg_heavyside(-x, -bnd2, e)
    F = A*B
    return F

def heavyside(x, bnd):
    """
    Function that construct a (singular) heavyside as function of x.
    
    Parameters
    ----------
    x : array of float
        Support of the function.
    bnd : float
        Jump position.
    
    Returns
    -------
    F : array of float
        (Singular) heavyside f(x).
    
    """
    F = np.zeros(len(x))
    F[np.where((x)>bnd)] = 1
    F[np.where((x)<bnd)] = 0
    return F

def step(x, bnd1, bnd2):
    """
    Function combining two (singular) heavyside to construct a (singular) step function.

    Parameters
    ----------
    x : array of float
        Support of the function.
    bnd1 : float
        First jump position.
    bnd2 : float
        Second jump position.

    Returns
    -------
    F : array of float
        (Singular) step function f(x).

    """
    A = heavyside(x, bnd1)
    B = heavyside(-x, -bnd2)
    F = A*B
    return F


#################################
#       Utilisation exemple     #
#################################

# if __name__=="__main__":

#     bnd = [0.7, 0.9]
#     xi = np.linspace(0, 1, 201)
    
#     singular_step = step(xi, bnd[0], bnd[1])
#     regularized_step = reg_step(xi, bnd[0], bnd[1], e=0.05)
    
#     nfig = 1
#     plt.figure() 
#     plt.title(r'$\mathrm{Regularized \ step \ function}$')                                             
#     plt.grid()                                                        
#     plt.plot(xi, singular_step,'--', label = r'$\mathrm{Singular \ step}$')      
#     plt.plot(xi, regularized_step, label = r'$\mathrm{Regularized \ step}$')      
    
#     plt.xlabel(r'$\xi$', fontsize = 12)                                            
#     plt.ylabel(r'$f(\xi)$', fontsize = 12)
#     plt.xlim((0,1))
#     plt.legend()

