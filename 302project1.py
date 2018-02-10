#for project 1 of PHY302, problem 5

from math import sinh, cosh, sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt



v = pi/3
a = np.linspace(-5,5,100)
y = np.linspace(-10,10,1000)
x = np.linspace(-10,10,1000)
x,y = np.meshgrid(x,y)

#I want to find the pairs of x,y that satisfy a hyperbolic equation for a given a-value,
#store all of those pairs in two arrays, then change a's value and repeat.
#What I would like to obtain would be essentially a list of lists for x and a list of lists for y
#each entry in both lists would correspond to the same a-value -- i.e., if I plugged
#in a=1 to start, then the 0th list in the x list of lists and the 0th entry in the y list of lists
# would both correspond to a=1
#So, above I generate a field of x and y's, meshgrid them together, then I would\
#use a for loop below to generate all the x and y lists for all the a values,
# and then I would contour to X-Y being equal to 1
for i in range(len(a)):
    X = (x/(a*cos(v)))**2
    Y = (y/(a*sin(v)))**2

plt.clf()
plt.contour(x,y,(X-Y),[1])
plt.show()
