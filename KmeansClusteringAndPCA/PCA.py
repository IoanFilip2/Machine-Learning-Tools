import sys, math

import scipy.io

import numpy as np

import matplotlib.pyplot as plt



### Loading and visualizing data set to be projected (dimensionality reduction)

filename = sys.argv[1]

the_data = scipy.io.loadmat(filename)['X']


# Converts data array with rows encoding gray-scale images into one image 
def displayData(array):

	[number, res] = np.shape(array)

	sizes = int(math.sqrt(res))
	num_size = int(math.sqrt(number))

	pad = 2

	horizontal = int(math.sqrt(number))

	vertical = int(number / horizontal)

	picture = np.zeros((1, (sizes + pad)*horizontal))

	for row in range(vertical):

		next_row = np.zeros((sizes + pad, 1))

		for column in range(horizontal):

			next = array[row * num_size + column, :].reshape(sizes, sizes)
			next = np.column_stack((next, np.zeros((sizes, pad))))
			next = np.row_stack((next, np.zeros((pad, sizes + pad))))

			next_row = np.column_stack((next_row, next))

		next_row = next_row[:, 1:]


		picture = np.row_stack((picture, next_row))


	picture = picture[1:, :].transpose()


	plt.imshow(picture, cmap='gray')
	#plt.savefig('test.png')
	plt.show()


## Displaying a specified portion of the data 
#displayData(the_data[0:100, :])


### Normalizing the features

mean = np.average(the_data, axis = 0)

difference = the_data - mean

st_dev = np.sum(difference * difference, axis = 0)/(np.shape(the_data)[0])

old_data = the_data

# Replacing the given data with its normalization
the_data = difference/st_dev


### Applying SVD on normalized data

# Matrix V contains as rows the feature eigenvalues from input data (e.g. 1024x1024 faces features) 
[U, S, V] = np.linalg.svd(the_data, full_matrices = True)

### Visualizing the most relevant eigen-directions for image features (e.g. faces)

num_features = 4

#displayData(V[0:num_features, :])

### Projecting data onto largest eigenspace (PCA)

num_dimensions = 100

proj_data = np.dot(the_data, V[0:num_dimensions, :].transpose())

# Recovering the projected data by taking the pre-image back to the original-size space
recover_data =  np.dot(proj_data, V[0:num_dimensions, :])


### Comparing projected data to the original input
displayData(recover_data[0:100, :])

displayData(old_data[0:100, :])