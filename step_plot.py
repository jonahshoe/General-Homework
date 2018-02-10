#heaviside function step plotting program

def heaviside(x):
   """Heaviside step function"""

   theta = None
   if x < 0:
      theta = 0.
   elif x == 0:
      theta = 0.5
   else:
      theta = 1.

   return theta


xList = []
thetaList = []

x = -4.0
i = 0

while i < 17:
    xList.append(x)
    thetaList.append(heaviside(x))
    i = i + 1
    x = x + 0.5

for j in range(len(xList)):
    print(xList[j] , '  ' , thetaList[j])

import matplotlib.pyplot as plt
plt.plot(xList, thetaList, '-o', color="red", linewidth=2)
plt.show()
