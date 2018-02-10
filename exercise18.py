#for exercise 18 in DF tutorial of phy302


from math import exp, pi, sin, sqrt, cos

#only N needs to be changed to change both limits of the sum
xList = []
x = -1
sigma = 0
sigmaList = []
N = 10000
lamda = 2

#sigmaN adds sigma to itself recursively with each n, which increments by 1
def sigmaN(x):
    n = -N
    global sigma
    sigma = 0
    while n <= N:
        sigma = sigma + 1 / lamda * cos(2 * pi * n / lamda * x)
        n += 1
    return sigma

#generates lists of x and sigma using the sigmaN function
while x <= 1:
    sigma = sigmaN(x)
    sigmaList.append(sigma)
    xList.append(x)
    x += 0.01


import matplotlib.pyplot as plt
plt.plot(xList, sigmaList, color="red", linewidth=2)
plt.show()
