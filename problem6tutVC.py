#for problem 6 of tutorial VC, PHY302

from math import sinh, cosh, sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



a = 1
v = np.linspace(-pi,pi,1000)
u = pi/3

X = a * np.sinh(v) / (np.cosh(v) - cos(u))
Y = a * sin(u) / (np.cosh(v)-cos(u))


Z = np.ones(1000)

for m in range(100):
    Z = m/100 * np.ones(1000)
    ax.plot(X,Y,Z, color='red')

plt.show()


U, V = np.meshgrid(np.arange(-1*np.pi, 1*np.pi, 0.1), np.arange(-1*np.pi, 1*np.pi, 0.1))
a = 1

X = a * np.sinh(V) / (np.cosh(V) - np.cos(U))
Y = a * np.sin(U) / (np.cosh(V)-np.cos(U))

Xhat = -np.sinh(V) * np.sin(U) / (np.cosh(V)-np.cos(U))
Yhat = (np.cos(U)*np.cosh(V) - 1) / (np.cosh(V)-np.cos(U))


fig = plt.figure()
#ax2 = fig.add_subplot(111)
#Q = ax2.quiver(Xhat,Yhat,X,Y, units='width')


Xhat1 = (1 - np.cos(U)*np.cosh(V)) / (np.cosh(V)-np.cos(U))
Yhat1 = -np.sin(U)**2 / (np.cosh(V)-np.cos(U))


#ax1 = fig.add_subplot(211)
#Q = plt.quiver(X,Y,Xhat1,Yhat1)
Q = plt.quiver(X,Y,Xhat,Yhat)
plt.xlim(-2,2), plt.ylim(-2,2)

plt.show()
