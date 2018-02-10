#for project 1 of PHY302, problem 5

from math import sinh, cosh, sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

U, V = np.meshgrid(np.arange(0, 2*np.pi, 0.1), np.arange(0, 2*np.pi, 0.1))
a = 1

X = a * np.cosh(U) * np.cos(V)
Y = a * np.sinh(U) * np.sin(V)

U1 = np.sinh(U)
V1 = np.sin(V)
R = np.sqrt(U1**2 + V1**2)
Xhat = U1 * np.cos(V) / R
Yhat = np.cosh(U) * V1 / R


fig = plt.figure()
#ax2 = fig.add_subplot(111)
#Q = ax2.quiver(Xhat,Yhat,X,Y, units='width')


Xhat1 = -np.cosh(U) * V1 / R
Yhat1 = U1 * np.cos(V) / R


#ax1 = fig.add_subplot(211)
Q = plt.quiver(Xhat1,Yhat1,X,Y)
Q = plt.quiver(Xhat,Yhat,X,Y)

plt.show()
