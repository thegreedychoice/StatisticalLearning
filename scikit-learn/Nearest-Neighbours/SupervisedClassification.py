"""
This performs Supervised Nearest Neighbor classification on the Iris Data Set into 3 classes.
"""

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

#Load the dataset
iris = datasets.load_iris()
#Load only the first 2 columns or features from the training dataset
X = iris.data[:,:2]
Y = iris.target


#Num of Neighbors
num_neighbors = 20

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#Step size in the mesh
h = 0.2
for w in ['uniform', 'distance']:
	#Create an instance of neighbours classifier
	clf = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, weights=w)
	#Fit the Training Data on the classifier
	clf.fit(X, Y)

	#Create a MeshGrid in order to plot the decision boundary. Find the boundary values of the Grid
	X1 = X[:,0]
	X2 = X[:,1]
	x_min, x_max = X1.min() - 1, X1.max() + 1
	y_min, y_max = X2.min() - 1, X2.max() + 1 
	xg, yg = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min, y_max,h))

	#xg and yg can be used as test data for the classifier
	#Concatenate the flattedned array and then generate a 2d array of new features
	Z = clf.predict(np.c_[xg.ravel(), yg.ravel()])

	#Put the result in the color plot
	Z = Z.reshape(xg.shape)

	plt.figure()
	plt.pcolormesh(xg, yg, Z, cmap=cmap_light)

	#plot the training points as well
	plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap_bold)
	plt.xlim(xg.min(), xg.max())
	plt.ylim(yg.min(), yg.max())
	plt.title("Iris 3-Class Neighbors Classification (k = %i | weights = %s"%(num_neighbors, w))


plt.show()


