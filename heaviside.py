# heaviside step function

x = 1
theta = None
if x < 0:
    theta = 0.
elif x == 0:
    theta = 0.5
else:
    theta = 1

print("Theta(" + str(x) + ") = " + str(theta))
