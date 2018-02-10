#for testing homework code examples




from math import sinh, cosh, sin, cos, pi

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')







a = 1

u = np.linspace(-2,2,1000)

v = pi/3



X = a * cos(v) * np.cosh(u)

Y = a * sin(v) * np.sinh(u)





Z = np.ones(1000)



for m in range(100):

    Z = m/100 * np.ones(1000)

    ax.plot(X,Y,Z)



plt.show()
