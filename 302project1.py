#for project 1 of PHY302, problem 5

from math import sinh, cosh, sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



a = 1
v = np.linspace(-pi,pi,1000)
u = pi/6

X = a * np.cos(v) * cosh(u)
Y = a * np.sin(v) * sinh(u)


Z = np.ones(1000)

for m in range(100):
    Z = m/100 * np.ones(1000)
    ax.plot(X,Y,Z, color='red')
plt.title('Constant u = pi/6 with a=0.1')
plt.xlabel('x'), plt.ylabel('y')
plt.show()
