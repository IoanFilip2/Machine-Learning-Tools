import sys, math

import numpy as np

import matplotlib.pyplot as plt

import scipy.optimize as op


## Read data from table

## DEGREEE PARAMETER for polynomial terms in region separation
## LAMBDA_param for regularization weight, if == 0, then have overfitting

# Assumes filename given as first input to the script:
filename = sys.argv[1]

my_data = np.genfromtxt(filename, delimiter = ',')

[r, c] = np.shape(my_data)

c_init = c

# Plotting 2D data with labels, only if X has two features
fig = plt.figure()

if c_init == 3:    
    red = my_data[abs(my_data[:, c-1] - 1)<0.01]
    blue = my_data[abs(my_data[:, c-1] - 1) >= 0.01]
    plt.plot(red[:, 0], red[:, 1], 'rx', label = 'admitted')
    plt.plot(blue[:, 0], blue[:, 1], 'bx', label = 'rejected')
    plt.xlabel("X-parameter")    
    plt.ylabel("Y-parameter")
    #plt.show()
    
# Reshaping the data, separating the y-label values from input
X = my_data[:, 0:c-1]
#X = np.column_stack((np.ones((r, 1)), X))
X = X.reshape(r, c-1)

X_init = X

y = my_data[:, c-1]
y.reshape(r, 1)

c = c - 1
### Introducing polynomial features for the regression:
degree = 6
##c = 2
#print(c)
#print(r)

all_powers = []
indexList = [[]]

length = 0

while (length < c):

	while(len(indexList[0]) == length):

		last_list = indexList.pop(0)
		for i in range(degree + 1):
			if sum(last_list) + i <= degree:
				new_list = last_list + [i]
				indexList.append(new_list)
	length += 1

for power in indexList:
	if power not in all_powers:
		all_powers.append(power)

print(np.shape(all_powers))

new_data = np.zeros((r, 1))

for feat_index in range(len(all_powers)):
	new_column = np.ones((r, 1))
	
	for index in range(c):
		new_column = new_column * np.power(X[:, index].reshape(r,1), all_powers[feat_index][index])


	new_data = np.column_stack((new_data, new_column))


#print(np.shape(new_data))
new_data = new_data.reshape(r, len(all_powers) + 1)

## Resizing the data:

[r, c] = np.shape(new_data)

# Removing the first zero column

X = new_data[:, 1:c]

c = c - 1
initialTheta = np.zeros((c, 1))


## OPTIMIZATION: Minimizing using built-in function, using Regularized Logistic Regression

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunction(X, y, theta, lambda_param):
    [r, c] = np.shape(X)
    tempProduct = X.dot(theta)
    tempVect1 = np.log(sigmoid(tempProduct))
    tempVect2 = np.log(1 - sigmoid(tempProduct))

    yVect2 = 1 - y

    temp = tempVect1*y + tempVect2*yVect2
    temp = temp.reshape(r, 1)
    return -1/r * np.ones((1, r)).dot(temp) + lambda_param/(2*r)*(np.sum(theta*theta))


def logisticGradient(X, y, theta, lambda_param):
    X = X.reshape(r,c)

    y = y.reshape(r,1)
    theta = theta.reshape(c, 1)

    interMultiply = (X.dot(theta)).reshape(r,1)
    grad = 1/r * (X.transpose()).dot(sigmoid(interMultiply) - y)

    reg_term = np.zeros((1,1))
    reg_factor  = (lambda_param/r) * theta[1:c, 0]

    reg_factor = reg_factor.reshape(c-1, 1)

    reg_term = np.row_stack((reg_term, reg_factor))
    reg_term = reg_term.reshape((c,1))

    grad = grad.reshape(c,1)
    
    return grad + reg_term

## Running the optimization:

# Fixing a regularization parameter:

lambda_param = 0.85

result = op.minimize(lambda t: costFunction(X, y, t, lambda_param), initialTheta, method = 'TNC', jac = lambda t: logisticGradient(X,y,t, lambda_param))
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

if c_init == 3:
	x_min = np.min(X_init[:, 0])
	x_max = np.max(X_init[:, 0])

	x_delta = (x_max - x_min)/250

	y_min = np.min(X_init[:,1])
	y_max = np.max(X_init[:,1])

	y_delta = (y_max - y_min)/250

	x = np.arange(x_min, x_max, x_delta)
	y = np.arange(y_min, y_max, y_delta)

	#print(np.shape(x))
	#print(np.shape(y))
    
	[X, Y] = np.meshgrid(x, y)
	Y_trans = Y.transpose()

	Z = np.zeros((250, 250))
    ## NEED separate method for feature mapping:

	for col_num in range(250):
	
		featureMappingX = np.column_stack((x.reshape(250, 1), (Y_trans[:, col_num]).reshape(250, 1)))
	
		degree = 6

		new_featuresX = np.zeros((250, 1))

		for feat_index in range(len(all_powers)):
			new_column = np.ones((250, 1))
	
			for index in range(2):
				new_column = new_column * np.power(featureMappingX[:, index].reshape(250,1), all_powers[feat_index][index])

			new_featuresX = np.column_stack((new_featuresX, new_column))

		#print(np.shape(new_featuresX))
		#print(len(all_powers) + 1)
		#new_featuresX = new_data.reshape(250, len(all_powers) + 1)


		# len(all_powers) == 28 here
		new_featuresX = new_featuresX[:, 1:len(all_powers) + 1]
		
		Z[:, col_num] = new_featuresX.dot(optimalTheta)

    ## Creating the plot
	
	#optimum = optimalTheta.flatten()
	#print(np.shape(new_featuresX))
	#print(np.shape(optimalTheta))
	#print(np.shape(Z))

	plt.contour(X, Y, Z, [0], colors = 'g')
	plt.title('Data separation')
	plt.show()
