"""
This program will complete the face i.e., predict the pixel values of lower half given the values of the
upper half. 
We will implement multi-output estimators to complete the images.
The first column of images shows true faces. The next columns illustrate how extremely randomized trees,
k nearest neighbors, linear regression and ridge regression complete the lower half of those faces.
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

#Load the faces dataset
data = fetch_olivetti_faces()
#labels hold values (0-39) for 40 unique faces
labels = data.target

#data.images shape = (400,64,64)
#images shape = (400, 4096)
images = data.images.reshape((len(data.images), -1))
#len(train) = 300
train = images[labels < 30]
#len(test) = 90
test = images[labels > 30]

#Test on a small number of people
n_faces = 5
rng =  check_random_state(6)
#face_ids is a list of 5 random integar values <= 90
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids,:]

#Now we need to split the data/images into two parts each representing upper and lower parts of the faces.
n_pixels = images.shape[1] #4096
#get pixels for all the rows from columns 0 - 2048 (upper half of face)
X_train =  train[ :, :np.ceil(0.5 * n_pixels)]
y_train = train[ :, np.floor(0.5 * n_pixels):] #pixels from columns 2048 - 4096 (lower half of the face)
X_test = test[ : , :np.ceil(0.5 * n_pixels)]
y_test = test[ :, np.floor(0.5 * n_pixels):]


#Create the output Estimators and put in a dictionary
estimators = {
	"Extra Trees" : ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
	"K-nn" : KNeighborsRegressor(),
	"Linear Regression" : LinearRegression(),
	"Ridge" : RidgeCV(),
}


#Create a dictionary to store the predicted values of each estimator
y_predict = dict()
for name, estimator in estimators.items():
	y_predict[name] = estimator.fit(X_train, y_train).predict(X_test)

#plot the completed faces
#Shape of image_shape = (64,64)
image_shape =(data.images.shape[1], data.images.shape[2])


n_plot_cols = 5
n_plot_rows = n_faces
plt.figure(figsize=(2. * n_plot_cols, 2.26 * n_plot_rows))
plt.suptitle("Face Completion with Multiple Regression EstimatorsFace Completion with Multiple Regression Estimato, size=16)

for i in range(n_faces):
	#Combine the original upper and lower halves of the face
	#true face will have shape (4096,)
	true_face = np.hstack((X_test[i], y_test[i]))
	if i:
		sub = plt.subplot(n_plot_rows, n_plot_cols, i*n_plot_cols+1)
	else:
		sub = plt.subplot(n_plot_rows, n_plot_cols, i*n_plot_cols+1, title="true faces")

	sub.axis("off")
	sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")

	#Plot the estimated faces of the corresponding true face/person for each estimator

	for j, est_name in enumerate(sorted(estimators)):
		complete_face = np.hstack((X_test[i], y_predict[est_name][i]))

		if i:
			sub = plt.subplot(n_plot_rows, n_plot_cols, i*n_plot_cols+2+j)
		else:
			sub = plt.subplot(n_plot_rows, n_plot_cols, i*n_plot_cols+2+j, title=est_name)

		sub.axis("off")
		sub.imshow(complete_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")


plt.show()












