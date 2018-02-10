#for project 1 of PHY302, problem 5

from math import sinh, cosh, sin, cos
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

def xfunk(u,v,a):
    return a*cosh(u)*cos(v)
def yfunk(u,v,a):
    return a*sinh(u)*sin(v)
def zfunk(z):
    return z

a = -1
u = 1
v = -10
z = -10
aList = np.arange(-1,1,0.1)
xList = np.array([])
yList = np.array([])
uList = np.arange(-10,10,0.1)
vList = np.arange(-10,10,0.1)
zListVar = np.arange(-10,10,0.1)


zList = np.ones(20)
zList = z*zList

def constU(v,a):
    x = a * cosh(u) * cos(v)
    y = a * sinh(u) * sin(v)
    return x, y



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xList, yList = np.meshgrid(xList, yList)
# Plot the surface.
surf = ax.plot_surface(xList, yList, zList, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
