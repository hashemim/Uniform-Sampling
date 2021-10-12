#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RANDOM UNITARY MATRICES
The code provides a uniform random sample of U(n) group. These are n \times n
square unitary matrices. We also show the uniformity for U(2) using the Bloch
sphere.
"""

""" First we need to produce an N by N complex random matrix whose real and 
 imaginary parts are drawn from Normal distribution """
 

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# WE can our own QR decomposition using Gram-Schmidt process.
def GSchmidt(V):
    m,N = V.shape
    R = np.zeros([N,N],dtype = 'complex_')
    Q = np.zeros([N,N],dtype = 'complex_')
    R[0,0] = linalg.norm(V[:,0])
    Q[:,0] = np.divide(V[:,0],R[0,0])
    for k in range(1,N):
        R[:k,k] = np.matmul(Q[:,:k].conj().T, V[:,k])
        Q[:,k] = V[:,k] - np.matmul(Q[:,:k],R[:k,k])
        R[k,k] = linalg.norm(Q[:,k])
        Q[:,k] = np.divide(Q[:,k],R[k,k])
    return Q,R
 
def Unrandom(N = 2):
    V = np.random.normal(0.0, 1.0, [N,N])+1j*np.random.normal(0.0, 1.0, [N,N])
    Q,R = linalg.qr(V)
    # WE can use our own function too Q, R = GSchmidt(V)
    return Q

# Here we will test the implementation for N=2 (U(2))
M = 10000
sx = np.array([[0,1], [1,0]])
sy = np.array([[0,-1j], [1j,0]])
sz = np.array([[1,0], [0,-1]])

# U = np.zeros([M,N,N])
coor= np.zeros([M,3])
for idx in range(M):
    Q = Unrandom(2)
    coor[idx,0] = np.real_if_close(np.linalg.multi_dot([Q.conj().T,sx,Q])[0,0])
    coor[idx,1] = np.real_if_close(np.linalg.multi_dot([Q.conj().T,sy,Q])[0,0])
    coor[idx,2] = np.real_if_close(np.linalg.multi_dot([Q.conj().T,sz,Q])[0,0])
    

# Creating figure
fig = plt.figure()
ax = fig.add_subplot(111,projection ="3d")
 
# Creating plot
ax.scatter3D(coor[:,0],coor[:,1],coor[:,2], color = "green", s = 0.1)
#ax.plot_wireframe(coor[:,0],coor[:,1],coor[:,2], color = "green")
ax.view_init(45,45)
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()
