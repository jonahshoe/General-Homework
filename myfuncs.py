#heaviside, fahrenheit to kelvin, kelvin to celsius functions


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




def k2c(T):
    """Kelvin to Celsius converter function"""
    cTemp = T - 273.15
    return cTemp


def f_to_k(theta):
    """Fahrenheit to Kelvin converter function"""
    return (theta - 32)*5/9


    
