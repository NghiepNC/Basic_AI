import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# random data
b = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]
A = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# Visualize data
plt.plot(A,b,'ro')

# Change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

# Create A square
x_square = np.array([A[:,0]**2]).T
A = np.concatenate((x_square, A), axis =1)

# Create vector 1
ones = np.ones((A.shape[0],1), dtype=np.int8)

# Combine 1 and A
A = np.concatenate((A, ones), axis =1)

# Use fomular
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)

# Test data to draw
x0 = np.linspace(1,25,10000)
y0 = x[0][0]*x0*x0+ x[1][0]*x0 + x[2][0]

plt.plot(x0,y0)

# Test predicting data
x_test = 12
y_test = x_test*x[0][0] + x[1][0]

print(y_test)
plt.show()