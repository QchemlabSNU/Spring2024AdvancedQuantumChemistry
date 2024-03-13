#!/usr/bin/env python
#
# Author: Seunghoon Lee (seunghoonlee@snu.ac.kr)
#

'''
Verification of orthonormality condition of an eigenfunction set
of time independent Schrodinger equation for a particle in an one-dimensional box
'''

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def eigenfunction(n, L):
    return lambda x: np.sqrt(2. / L) * np.sin(n * np.pi * x / L)

def eigenvalue(n, m, L, hbar):
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2. * m * L ** 2)

def innerproduct(x1, x2, bra, ket):
    integrand = lambda x: np.conj(bra(x)) * ket(x)
    result, _ = integrate.quad(integrand, x1, x2)
    return result

hbar = 1.0  # planck constant in atomic unit
L = 1.0     # size of box
m = 1.0     # mass of particle
n = 4       # number of eigenfunctions 

orthonormality_check = np.zeros((n,n))
for i in range(1, n+1):
    for j in range(1, n+1):
        orthonormality_check[i-1, j-1] = innerproduct(0, L, eigenfunction(i, L), eigenfunction(j, L))

print(orthonormality_check)

