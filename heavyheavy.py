#for graphing heaviside functions in math methods

from math import exp, pi, sin

def heaviside(x):
    """Heaviside step function"""

    theta = None
    if x < 0:
        theta = 0.
    elif x == 0:
        theta = None
    else:
        theta = 1

    return theta

tList = []
tStart = -1
tEnd = 1
heavyOut = []
heavy = None
while tStart <= tEnd:
    if tStart != 0:
        tList.append(tStart)
        heavy = exp(-tStart) * heaviside(tStart)
        heavyOut.append(heavy)
    tStart += 0.01



import matplotlib.pyplot as plt
plt.plot(tList, heavyOut, '-o', color="red", linewidth=2)
plt.show()
