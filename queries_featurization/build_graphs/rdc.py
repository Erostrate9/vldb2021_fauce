"""
http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
"""
import argparse
import numpy as np
from scipy.stats import rankdata
import csv
import pandas

def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))

def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser("rdc")
    args.add_argument("-i","--input", default = 'full_data_12_19_new_CHKK_cluster1.csv',
                      help="Input table data")
    args.add_argument("-o","--output", default = 'matrix_12_19_CHKK_cluster1.csv',
                      help="Ouput rdc matrix of given table")
    # args.add_argument("-k","--random_projections", default = 20,
    #                   help="number of random projections to use")
    # args.add_argument("-s","--scale_parameter", default = 1/6.,
    #                   help="scale parameter")
    # args.add_argument("-n","--compute_times", default = 1,
    #                   help="number of times to compute the RDC and return the median (for stability)")
    return args.parse_args()

def main(args):
  df=pandas.read_csv(args.input, escapechar='\\')
  array=df.values
  matrix=np.zeros((len(array[0]), len(array[0])))
  for i in range(len(array[0])):
      for j in range(len(array[0])):
          matrix[i][j]=rdc(array[:,i], array[:,j])
  my_df=pandas.DataFrame(matrix)
  my_df.to_csv(args.output, index=False, header=False)


if __name__=="__main__":
    args = parse_args()
    main(args)