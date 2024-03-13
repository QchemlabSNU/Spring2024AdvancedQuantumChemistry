#!/usr/bin/env python
#
# Author: Seunghoon Lee (seunghoonlee@snu.ac.kr)
#

'''
Visualizing an eigenfunction of time independent Schrodinger equation
for a particle in an one-dimensional box
'''

import numpy as np
import matplotlib.pyplot as plt

def eigenfunction(x, n, L):
    return np.sqrt(2. / L) * np.sin(n * np.pi * x / L)

def eigenvalue(n, m, L, hbar):
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2. * m * L ** 2)

hbar = 1.0  # planck constant in atomic unit
L = 1.0     # size of box
m = 1.0     # mass of particle
n = 1       # quantum number
x = np.linspace(0, L, 50)
psi_n = eigenfunction(x, n, L) 
e_n = eigenvalue(n, m, L, hbar) 

plt.axvline(0, color='black', linewidth=5)   # L wall
plt.axvline(L, color='black', linewidth=5)   # R wall
plt.plot(x, psi_n, label=f'n = %d, energy = %f' % (n, e_n))
plt.xlabel('x')
plt.ylabel('$\\psi_n(x)$')
plt.title('Particle in a Box')
plt.legend()
plt.show()

