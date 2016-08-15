import sys, math

from scipy import misc

import numpy as np

import matplotlib.pyplot as plt

### Performs K-means clustering for image compression
### Step-by-step algo can be applied to any input data

## THE MAIN CLUSTERING FUNCTIONS:
### TO DO: RUNS LINEARLY, HOW TO OPTIMIZE???

# Random initialization of clusters, chooses num_clusters, possibly non-distinct ones
def kMeansInitCentroids(img_vector, num_centroids):

	dimensions = np.shape(img_vector)

	centroids = []

	for i in range(num_centroids):
		point = img_vector[np.random.choice(range(dimensions[0])), :]
		
		#if np.array(point) not in centroids:
		centroids.append(point)

	return centroids


def runKMeans(img_vector, init_centroids, max_iterations):

	centroids = init_centroids

	for i in range(max_iterations):
		indexes = findClosestCentroids(img_vector, centroids)
		centroids = averageCentroids(img_vector, indexes, len(centroids))

	return [centroids, indexes]


def findClosestCentroids(img_vector, centroids):

	dimensions = np.shape(img_vector)

	indexes = []

	for i in range(dimensions[0]):
		min_dist = np.sum(  (img_vector[i, :] - centroids[0])*(img_vector[i, :] - centroids[0]) )
		min_index = 0

		for j in range(1, len(centroids)):
			if( np.sum(  (img_vector[i, :] - centroids[j])*(img_vector[i, :] - centroids[j]) ) < min_dist ):
				min_dist = np.sum(  (img_vector[i, :] - centroids[j])*(img_vector[i, :] - centroids[j]))
				min_index = j


		indexes.append(min_index)

	return indexes


def averageCentroids(img_vector, indexes, num_centroids):

	centroids = []

	for i in range(num_centroids):

		indexes = np.array(indexes)
		average = sum(img_vector[indexes == i])
		average = average/(len(img_vector[indexes == i]))

		centroids.append(average)

	return centroids


## Getting the data from specific input, SHOULD be a .png FILE
filename = sys.argv[1]

init_picture = misc.imread(filename)
img_data = init_picture

# size is a vector with 3 entries: rows, columns and num_colors of 'rgb' == 3
size = np.shape(img_data)


# Normalizing the pixel entries so they lie in [0,1]:
img_data = img_data/255


# Reshaping the image matrix as a long vector:
img_vector = img_data.reshape(size[0]*size[1], 3)


## Initializations:

num_clusters = 16
max_iterations = 10

# Random initialzation of the centroids:
init_centroids = kMeansInitCentroids(img_vector, num_clusters)


## Running the K-means algorithm with given initializations:
[centroids, indexes] = runKMeans(img_vector, init_centroids, max_iterations)

## Outputting the cluster members:
cluster_indexes = findClosestCentroids(img_vector, centroids)


## Recovering an image from clusters:
centroids = np.array(centroids)
recover_vector = centroids[cluster_indexes, :]

## Reshaping the recovered image into rectangular pixels:
recover_img  = recover_vector.reshape(size[0], size[1], 3)


### Plotting the compressed and initial images side by side
fig = plt.figure()

I=fig.add_subplot(1,2,1)
imgplot = plt.imshow(init_picture)
I.set_title('Original image')

I=fig.add_subplot(1,2,2)
imgplot = plt.imshow(recover_img)
I.set_title('Compressed image with '+str(num_clusters)+' color clusters')

plt.show()