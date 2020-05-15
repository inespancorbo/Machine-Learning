# Matplotlib




# NumPy
- Reading csv files to numpy arrays: np.genfromtxt("./path/to/file", delimiter=",")
- Methods:
  - shape, dtype



# Pandas


# Scikit-learn


# SciPy

### from scipy import linalg
- scipy.linalg (or use numpy.linalg) `https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html`
  - scipy.linalg operations can be applied equally to numpy.matrix or to 2D numpy.ndarray objects
    - np.mat('[1 2;3 4]') vs. np.array([[1,2],[3,4]]) (np.mat is discouraged)
  - Basic routines (inverse, determinant, condition number, norms, multiplication), Decompositions (SVD, eigen), etc.

### from scipy import stats


### from scipy.spatial import distances
- Distance functions between two numeric vectors u and v:
  - euclidean/sqeuclidean distance: computes the Euclidean/Euclidean squared distance between two 1-D arrays.
  - chebyshev distance: computes the Chebyshev distance between two 1-D arrays u and v, which is defined as max<sub>i</sub> |u<sub>i</sub> - v<sub>i</sub>|.
  - minkowski distance: think real analysis l^p, p-norm.
  - cosine distance (1 - cos angle formed by u and v), correlation distance (1 - correlation between u and v, etc.
- Distance matrix computation from a collection of raw observation vectors stored in a rectangular array (note you could also broadcast, might be painful):
  - pdist(X[, metric]): Pairwise distances between observations in n-dimensional space.
  - cdist(XA, XB[, metric]): Compute distance between each pair of the two collections of inputs.





