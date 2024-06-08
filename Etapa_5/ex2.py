from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt 
def R(x):
    return 1.0/(1.0 + 25.0*x**2)
# Interpolating points
xi = np.array([-5, -4.5, -3.8, -3, -2.07, -1.45, -1, -0.72, -0.51,-0.43,
-0.26, 0, 0.17, 0.33, 0.5, 0.7, 1, 1.5, 2, 3, 4, 4.5, 5])
yi = R(xi)
# Define a set of points to evaluate the functions
xeval = np.linspace(-5, 5, 2000)
yeval = R(xeval)
# Compute the piecewise liner polynomial
ylin = interp1d(xi, yi, kind='linear')
# Plot everything
plt.plot(xi, yi,'ob',
xeval, yeval, '-r',
xeval, ylin(xeval), '-g')

plt.show()

# interpolação quadrática
ylin = interp1d(xi, yi, kind='quadratic')
# Plot everything
plt.plot(xi, yi,'ob',
xeval, yeval, '-r',
xeval, ylin(xeval), '-g')

plt.show()

# interpolação cúbica
ylin = interp1d(xi, yi, kind='cubic')
# Plot everything
plt.plot(xi, yi,'ob',
xeval, yeval, '-r',
xeval, ylin(xeval), '-g')
plt.show()