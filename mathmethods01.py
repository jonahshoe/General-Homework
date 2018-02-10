#for math methods dirac delta homework
#imports quad integral function
from scipy.integrate import quad
from math import exp, pi

#epsilon is defined globally so it only needs to be changed here to change it everywhere
epsilon = 1

def diracIntegral(x):
    """This function takes the integral of the dirac delta function from x = -10 to 10"""
    return 1 / 3.1415 * epsilon / (x**2+epsilon**2)

#this uses the diracIntegral function to take the integral and assign it to the variable answer
#the error is because the python module or whatever used finds an error margin for the integral taken
#if you just put answer = quad(blah blah blah) then both the integral value and the error value
#are assigned to the answer variable as a list
answer, error = quad(diracIntegral, -10, 10)

print('The integral for epsilon = ', epsilon, ' is ', answer)


def dirac(xList):
    """Takes a population of values in xList and populates a sigmaList with the sigma from each x"""
    sigma = float()
    for i in range(len(xList)):
        sigma = 1 / 3.1415 * epsilon / (xList[i]**2+epsilon**2)
        sigmaList.append(sigma)




sigmaList = []
x = -10
xList = []

#populates the xList list. I used a step size of 0.0001. More and it becomes hard to see how thin
#the dirac delta function spike becomes with low epsilons
while x <= 10:
    xList.append(x)
    x += 0.0001

#calls the dirac function defined earlier and gives it the xList list to use to populate sigmaList
dirac(xList)

#plots the resulting sets of values
import matplotlib.pyplot as plt
plt.plot(xList, sigmaList, '-o', color="red", linewidth=2)
plt.show()

def diracWithFunction(x):
    return (x**2+1) * exp(-x**2) * eps / (x**2 + eps**2)


nList = []
epsList = []
#answerList = []
eps = 1
while eps >= 0.0001:
    epsList.append(eps)
    eps = eps * 9 / 10


for h in range(len(epsList)):
    eps = epsList[h]
    answer2, error2 = quad(diracWithFunction, -10, 10)
    #answerList.append(answer2)
    n = 1 / answer2
    nList.append(n)

plt.plot(nList, epsList, '-o', color="red", linewidth=2)
plt.axvline(1 / pi)
plt.show()
