#!/usr/bin/env python
#
# Author: Seunghoon Lee (seunghoonlee@snu.ac.kr)
#

'''
Visualizing an eigenfunction of time independent Schrodinger equation
for a particle in an one-dimensional box
V = - epsilon * sin (pi * x / L) for 0 < x < L and V = infty for x <= 0 or x >= L
'''

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def eigenfunction(n, L):
    return lambda x: np.sqrt(np.pi / L) * np.sin(n * np.pi * x / L)

def eigenvalue(n, m, L, hbar):
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2. * m * L ** 2)

def expectation(x1, x2, bra, op, ket):
    integrand = lambda x: np.conj(bra(x)) * op(x) * ket(x)
    result, _ = integrate.quad(integrand, x1, x2)
    return result

hbar = 1.0     # planck constant in atomic unit
L = 1.0        # size of box
m = 1.0        # mass of particle
n = 3          # number of eigenfunctions
epsilon = 0.1  # V(x) = - epsilon * sin (pi * x / L) for 0 < x < L
Vx = lambda x: - epsilon * np.sin(np.pi * x / L)  

Hmat = np.zeros((n, n))
for i in range(1, n+1):
    psi0_i = eigenfunction(i, L) 
    for j in range(1, n+1):
        psi0_j = eigenfunction(j, L) 
        if i == j:
            Hmat[i-1, j-1] += eigenvalue(i, m, L, hbar) 
        Hmat[i-1, j-1] += expectation(0, L, psi0_i, Vx, psi0_j)

val, vec = np.linalg.eig(Hmat)

idx = np.argsort(val)
val = val[idx]
vec = vec[:, idx]

print('Ground state energy')
print(val[0])
print('Ground state expansion coefficients')
print(vec[:, 0])

