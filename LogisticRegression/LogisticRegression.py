import sys, math

import numpy as np

import matplotlib.pyplot as plt

import scipy.optimize as op


## Read data from table

# Assumes filename given as first input to the script:
filename = sys.argv[1]

my_data = np.genfromtxt(filename, delimiter = ',')

[r, c] = np.shape(my_data)


# Plotting 2D data with labels, only if X has two features
fig = plt.figure()

if c == 3:    
    red = my_data[abs(my_data[:, c-1] - 1)<0.01]
    blue = my_data[abs(my_data[:, c-1] - 1) >= 0.01]
    plt.plot(red[:, 0], red[:, 1], 'rx', label = 'admitted')
    plt.plot(blue[:, 0], blue[:, 1], 'bx', label = 'rejected')
    plt.xlabel("X-parameter")    
    plt.ylabel("Y-parameter")
    #plt.show()
    


# Reshaping the data, separating the y-label values from input
X = my_data[:, 0:c-1]
X = np.column_stack((np.ones((r, 1)), X))
X = X.reshape(r, c)

y = my_data[:, c-1]
y.reshape(r, 1)

initialTheta = np.zeros((c, 1))

# Minimizing using built-in function:

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunction(X, y, theta):
    [r, c] = np.shape(X)
    tempProduct = X.dot(theta)
    tempVect1 = np.log(sigmoid(tempProduct))
    tempVect2 = np.log(1 - sigmoid(tempProduct))

    yVect2 = 1 - y

    temp = tempVect1*y + tempVect2*yVect2
    temp = temp.reshape(r, 1)
    return -1/r * np.ones((1, r)).dot(temp)


def logisticGradient(X, y, theta):
    X = X.reshape(r,c)

    y = y.reshape(r,1)
    theta = theta.reshape(c, 1)

    interMultiply = (X.dot(theta)).reshape(r,1)
    grad = 1/r * (X.transpose()).dot(sigmoid(interMultiply) - y)
    
    return grad.reshape(c,1)

# Running the optimization:
result = op.minimize(lambda t: costFunction(X, y, t), initialTheta, method = 'TNC', jac = lambda t: logisticGradient(X,y, t))
optimalTheta = result.x


### Plotting theta:
### CONTOUR PLOTS FOR 2D DATA

print("The optimal parameters are: ")
print(optimalTheta)

#bounds1 = np.arange( math.floor(np.min(X[:,1])), math.ceil(np.max(X[:, 1])) + 1) 
#images = 1/optimalTheta[2] *(-optimalTheta[0] - optimalTheta[1]*bounds1)

## Plotting the 2D data and regression line:
#if c == 3:
#    plt.plot(bounds1, images, 'g-', label = 'separator')
#    plt.legend()
#    plt.show()

if c == 3:
    x_min = np.min(X[:, 1])
    x_max = np.max(X[:, 1])

    x_delta = (x_max - x_min)/250

    y_min = np.min(X[:,2])
    y_max = np.max(X[:,2])

    y_delta = (y_max - y_min)/250

    x = np.arange(x_min - 1, x_max + 1, x_delta)
    y = np.arange(y_min - 1, y_max + 1, y_delta)
    
    [X, Y] = np.meshgrid(x, y)   

    optimum = optimalTheta.flatten()
    Z = optimum[0] + X*optimum[1] + Y*optimum[2]

    plt.contour(X, Y, Z, [0], colors = 'g')
    plt.title('Data separation')
    plt.show()
    







