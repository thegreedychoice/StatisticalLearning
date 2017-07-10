"""
Demonstrate the resolution of a regression problem using a k-Nearest Neighbor and the interpolation of the target using both barycenter and constant weights.
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

#Generate the sample data first
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
#Generate some test data 
Z = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

#Num of Neighbors
n_neighbors = 5
#Add noise to the labels 
#Add random noise to every 5th output value for the data
y[::5] += 1 * (0.5 - np.random.rand(8))

#Now perform Regression on the generated sample data
#Fit the regression Model

for i, w in enumerate(['uniform', 'distance']):
	knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=w)
	y_ = knn.fit(X, y).predict(Z)

	#plot the model on graph
	plt.subplot(2,1,i+1)
	plt.scatter(X, y, c='k', label='data')
	plt.plot(Z, y_, c='g', label='prediction')
	plt.axis('tight')
	plt.legend()
	plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                w))
plt.show()



