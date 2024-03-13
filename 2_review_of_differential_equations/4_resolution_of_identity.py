#!/usr/bin/env python
#
# Author: Seunghoon Lee (seunghoonlee@snu.ac.kr)
#

'''
Demonstration of resolution of identity using an eigenfunction set
of time independent Schrodinger equation for a particle in an one-dimensional box
'''

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)

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
n = 10     # number of eigenfunctions 

# Resolution of Identity
# |Psi> = \sum_n |psi_n> <psi_n|Psi> = \sum_n c_n * |psi_n>   
# where c_n = <psi_n|Psi>

Psi = lambda x: np.sqrt(1. / L) + x * 0

npt = 50
x = np.linspace(0, L, npt)

Psi_RI = np.zeros((npt))
for i in range(1, n+1):
    c_n = innerproduct(0, L, eigenfunction(i, L), Psi)
    Psi_RI_n = lambda x: c_n * eigenfunction(i, L)(x) 
    Psi_RI += Psi_RI_n(x)

plt.axvline(0, color='black', linewidth=5)   # L wall
plt.axvline(L, color='black', linewidth=5)   # R wall

plt.plot(x, Psi(x), label=f'Psi')
plt.plot(x, Psi_RI, label=f'Psi RI with # of basis = %d' % (n))

plt.xlabel('x')
plt.ylabel('$\\Psi(x)$')
plt.title('Particle in a Box')
plt.legend()
plt.show()

