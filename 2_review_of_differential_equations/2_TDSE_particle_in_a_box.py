#!/usr/bin/env python
#
# Author: Seunghoon Lee (seunghoonlee@snu.ac.kr)
#

'''
Visualizing time dependent stationary wave function
for a particle in an one-dimensional box
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def eigenfunction(x, n, L):
    return np.sqrt(2. / L) * np.sin(n * np.pi * x / L)

def eigenvalue(n, m, L, hbar):
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2. * m * L ** 2)

def phasefactor(En, t, hbar):
    return np.exp(-1j * En * t / hbar)

hbar = 1.0  # planck constant in atomic unit
L = 1.0     # size of box
m = 1.0     # mass of particle
n = 1       # quantum number
x = np.linspace(0, L, 50)
psi_n = eigenfunction(x, n, L) 
e_n = eigenvalue(n, m, L, hbar) 

fig, ax = plt.subplots()
line, = ax.plot(x, np.real(psi_n), label='Real part')
line_imag, = ax.plot(x, np.imag(psi_n), label='Imaginary part')
ax.set_xlim(-0.2, L+0.2)
ax.set_ylim(-1.5, 1.5)

plt.axvline(0, color='black', linewidth=5)   # L wall
plt.axvline(L, color='black', linewidth=5)   # R wall
plt.xlabel('x')
plt.ylabel('$\\psi_n(x, t)$')
plt.title(f'Particle in a Box with Time Evolution')
plt.legend()

def update(frame):
    t = frame / 100
    psi_t = psi_n * phasefactor(e_n, t, hbar)
    line.set_ydata(np.real(psi_t))
    line_imag.set_ydata(np.imag(psi_t))
    return line, line_imag,

ani = FuncAnimation(fig, update, frames=np.arange(0, 200), blit=True)
plt.show()

