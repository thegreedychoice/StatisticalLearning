from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = [[2, 1]]
nbrs = NearestNeighbors(n_neighbors=3, radius=1.5, algorithm='auto')
nbrs.fit(X)


r_distances, r_indices = nbrs.radius_neighbors(Y)
r_graph = nbrs.radius_neighbors_graph(Y)

print "Training Data (X)"
print X, '\n'

print "Test Data (Y)"
print Y, '\n'

k_distances, k_indices = nbrs.kneighbors(
    Y, n_neighbors=None, return_distance=True)
print "k-Neighbour Distances"
print k_distances, '\n'

print "k-Neighbour Indices"
print k_indices, '\n'


print "k-Neighbour Sparse Graph"
k_graph = nbrs.kneighbors_graph(Y, n_neighbors=None, mode='connectivity')
print k_graph, '\n'

print "k-Neighbour Sparse Connections"
print k_graph.toarray(), '\n'


r_distances_indices = nbrs.radius_neighbors(Y, radius=None, return_distance=True)
print "radius-Neighbour Distances"
print r_distances_indices[0][0], '\n'

print "radius-Neighbour Indices"
print r_distances_indices[1][0], '\n'


print "radius-Neighbour Sparse Graph"
r_graph = nbrs.radius_neighbors_graph(Y, radius=None, mode='distance')
print r_graph, '\n'

print "radius-Neighbour Sparse Connections"
print r_graph.toarray(), '\n'

print "Parameters of the Estimator :NearestNeighbors"
print nbrs.get_params(), '\n'



"""
Output :

Training Data (X)
[[-1 -1]
 [-2 -1]
 [-3 -2]
 [ 1  1]
 [ 2  1]
 [ 3  2]] 

Test Data (Y)
[[2, 1]] 

k-Neighbour Distances
[[ 0.          1.          1.41421356]] 

k-Neighbour Indices
[[4 3 5]] 

k-Neighbour Sparse Graph
  (0, 4)	1.0
  (0, 3)	1.0
  (0, 5)	1.0 

k-Neighbour Sparse Connections
[[ 0.  0.  0.  1.  1.  1.]] 

radius-Neighbour Distances
[ 1.          0.          1.41421356] 

radius-Neighbour Indices
[3 4 5] 

radius-Neighbour Sparse Graph
  (0, 3)	1.0
  (0, 4)	0.0
  (0, 5)	1.41421356237 

radius-Neighbour Sparse Connections
[[ 0.          0.          0.          1.          0.          1.41421356]] 

Parameters of the Estimator :NearestNeighbors
{'n_neighbors': 3, 'n_jobs': 1, 'algorithm': 'auto', 'metric': 'minkowski', 'metric_params': None, 'p': 2, 'radius': 1.5, 'leaf_size': 30} 


"""
