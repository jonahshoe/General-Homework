#requests an absolute temperature in kelvin and a temperature difference in fahrenheit
#from the user, then converts the temperature difference into kelvin and adds
#it to the kelvin temperature input, and prints out the sum

kelvin = float(input("Absolute temperature in Kelvin?: "))
#absolute temp cannot be negative:
while kelvin < 0:
    kelvin = float(input("Absolute temperature cannot be negative. Enter again: "))
dTheta = float(input("Temperature difference in degrees Fahrenheit?: "))

newKelvin = kelvin + 5 / 9 * dTheta #mathematical expression to compute the sum of kelvin and dTheta

#if the user enters a highly negative dTheta that makes the Kelvin temperature negative:
while newKelvin < 0:
    dTheta = float(input("Your temperature difference resulted in a negative absolute temperature. Enter again: "))
    newKelvin = kelvin + 5 / 9 * dTheta
else:
    print("The sum is ", newKelvin, " K")
