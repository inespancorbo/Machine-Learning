# Matplotlib




# NumPy
- Reading csv files to numpy arrays: np.genfromtxt("./path/to/file", delimiter=",")
- Methods:
  - shape, dtype



# Pandas


# Scikit-learn


# SciPy

### from scipy import linalg
- scipy.linalg (do not use numpy.linalg) `https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html`
  - scipy.linalg operations can be applied equally to numpy.matrix or to 2D numpy.ndarray objects
    - np.mat('[1 2;3 4]') vs. np.array([[1,2],[3,4]]) (np.mat is discouraged)
  - Basic routines (inverse, determinant, condition number, norms, multiplication), Decompositions (SVD, eigen), etc.

### from scipy import stats


### from scipy.spatial import distances
- Useful distance computations:
  - euclidean distance: computes the Euclidean distance between two 1-D arrays
  - chebyshev distance: computes the Chebyshev distance between two 1-D arrays $u$ and $v$, which is defined as $\max{i}|u_i-v_i|$
  - minkowski distance: 
  - h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x




