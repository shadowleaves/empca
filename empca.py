#!/usr/bin/env python

"""
Weighted Principal Component Analysis using Expectation Maximization

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nvar, nobs] and weights[nvar, nobs],

    m = empca(data, weights, options...)

Returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )

For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #  but no weights or even options...

Stephen Bailey, Spring 2012
"""

import numpy as N
import sys
#from scipy.sparse import dia_matrix
import scipy.sparse.linalg
import math
from random import sample


class Model(object):

    """
    A wrapper class for storing data, eigenvectors, and coefficients.

    Returned by empca() function.  Useful member variables:
      Inputs:
        - eigvec [nvec, nobs]
        - data   [nvar, nobs]
        - weights[nvar, nobs]

      Calculated from those inputs:
        - coeff  [nvar, nvec] - coeffs to reconstruct data using eigvec
        - model  [nvar, nobs] - reconstruction of data using eigvec,coeff

    Not yet implemented: eigenvalues, mean subtraction/bookkeeping
    """

    def __init__(self, eigvec, data, weights):
        """
        Create a Model object with eigenvectors, data, and weights.

        Dimensions:
          - eigvec [nvec, nobs]  = [k, j]
          - data   [nvar, nobs]  = [i, j]
          - weights[nvar, nobs]  = [i, j]
          - coeff  [nvar, nvec]  = [i, k]
        """
        self.eigvec = eigvec
        self.nvec = eigvec.shape[0]

        self.set_data(data, weights)

    def set_data(self, data, weights):
        """
        Assign a new data[nvar,nobs] and weights[nvar,nobs] to use with
        the existing eigenvectors.  Recalculates the coefficients and
        model fit.
        """
        self.data = data
        self.weights = weights

        self.nvar = data.shape[0]
        self.nobs = data.shape[1]
        self.coeff = N.zeros((self.nvar, self.nvec))
        self.model = N.zeros(self.data.shape)

        # - Calculate degrees of freedom
        ii = N.where(self.weights > 0)
        self.dof = self.data[ii].size - \
            self.eigvec.size - self.nvec * self.nvar

        #  Cache variance of unmasked data
        self._unmasked = ii
        self._unmasked_data_var = N.var(self.data[ii])

        self.solve_coeffs()

    def solve_coeffs(self):
        """
        Solve for c[i,k] such that data[i] ~= Sum_k: c[i,k] eigvec[k]
        """
        for i in range(self.nvar):
            #  Only do weighted solution if really necessary
            if N.any(self.weights[i] != self.weights[i, 0]):
                self.coeff[i] = _solve(
                    self.eigvec.T, self.data[i], self.weights[i])
            else:
                self.coeff[i] = N.dot(self.eigvec, self.data[i])

        self.solve_model()

    def solve_eigenvectors(self, smooth=None):
        """
        Solve for eigvec[k,j] such that data[i] = Sum_k: coeff[i,k] eigvec[k]
        """

        #  Utility function; faster than numpy.linalg.norm()
        def norm(x):
            return N.sqrt(N.dot(x, x))

        #  Make copy of data so we can modify it
        data = self.data.copy()

        #  Solve the eigenvectors one by one
        for k in range(self.nvec):

            #  Can we compact this loop into numpy matrix algebra?
            c = self.coeff[:, k]
            for j in range(self.nobs):
                w = self.weights[:, j]
                x = data[:, j]
                # self.eigvec[k, j] = c.dot(w*x) / c.dot(w*c)
                # self.eigvec[k, j] = w.dot(c*x) / w.dot(c*c)
                cw = c * w
                self.eigvec[k, j] = x.dot(cw) / c.dot(cw)

            if smooth is not None:
                self.eigvec[k] = smooth(self.eigvec[k])

            #  Remove this vector from the data before continuing with next
            #  Alternate: Resolve for coefficients before subtracting?
            #  Loop replaced with equivalent N.outer(c,v) call (faster)
            # for i in range(self.nvar):
            #     data[i] -= self.coeff[i,k] * self.eigvec[k]

            data -= N.outer(self.coeff[:, k], self.eigvec[k])

        #  Renormalize and re-orthogonalize the answer
        self.eigvec[0] /= norm(self.eigvec[0])
        for k in range(1, self.nvec):
            for kx in range(0, k):
                c = N.dot(self.eigvec[k], self.eigvec[kx])
                self.eigvec[k] -= c * self.eigvec[kx]

            self.eigvec[k] /= norm(self.eigvec[k])

        #  Recalculate model
        self.solve_model()

    def solve_model(self):
        """
        Uses eigenvectors and coefficients to model data
        """
        for i in range(self.nvar):
            self.model[i] = self.eigvec.T.dot(self.coeff[i])

    def chi2(self):
        """
        Returns sum( (model-data)^2 / weights )
        """
        delta = (self.model - self.data) * N.sqrt(self.weights)
        return N.sum(delta**2)

    def rchi2(self):
        """
        Returns reduced chi2 = chi2/dof
        """
        return self.chi2() / self.dof

    def _model_vec(self, i):
        """Return the model using just eigvec i"""
        return N.outer(self.coeff[:, i], self.eigvec[i])

    def R2vec(self, i):
        """
        Return fraction of data variance which is explained by vector i.

        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """

        d = self._model_vec(i) - self.data
        return 1.0 - N.var(d[self._unmasked]) / self._unmasked_data_var

    def R2(self, nvec=None):
        """
        Return fraction of data variance which is explained by the first
        nvec vectors.  Default is R2 for all vectors.

        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """
        if nvec is None:
            mx = self.model
        else:
            mx = N.zeros(self.data.shape)
            for i in range(nvec):
                mx += self._model_vec(i)

        d = mx - self.data

        #  Only consider R2 for unmasked data
        return 1.0 - N.var(d[self._unmasked]) / self._unmasked_data_var


def _random_orthonormal(nvec, nobs, seed):
    """
    Return array of random orthonormal vectors A[nvec, nobs]

    Doesn't protect against rare duplicate vectors leading to 0s
    """

    if seed is not None:
        N.random.seed(seed)

    A = N.random.normal(size=(nvec, nobs))
    for i in range(nvec):
        A[i] /= N.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= N.dot(A[j], A[i]) * A[j]
            A[i] /= N.linalg.norm(A[i])

    return A


def _solve(A, b, w):
    """
    Solve Ax = b with weights w; return x

    A : 2D array
    b : 1D array length A.shape[0]
    w : 1D array same length as b
    """

    #  Apply weights
    # nobs = len(w)
    # W = dia_matrix((w, 0), shape=(nobs, nobs))
    # bx = A.T.dot( W.dot(b) )
    # Ax = A.T.dot( W.dot(A) )

    b = A.T.dot(w * b)
    A = A.T.dot((A.T * w).T)

    if isinstance(A, scipy.sparse.spmatrix):
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        # x = N.linalg.solve(A, b)
        x = N.linalg.lstsq(A, b)[0]

    return x


# ------------------------------------------------------------------------
def empca(data, weights=None, niter=25, nvec=5, smooth=0,
          randseed=1, silent=False):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights

    Input:
      - data[nvar, nobs]
      - weights[nvar, nobs]

    Optional:
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - randseed : random number generator seed; None to not re-initialize

    Returns Model object
    """

    if weights is None:
        weights = N.ones(data.shape)

    if smooth > 0:
        smooth = SavitzkyGolay(width=smooth)
    else:
        smooth = None

    #  Basic dimensions
    nvar, nobs = data.shape
    assert data.shape == weights.shape

    #  degrees of freedom for reduced chi2
    ii = N.where(weights > 0)
    dof = data[ii].size - nvec * nobs - nvec * nvar

    #  Starting random guess
    eigvec = _random_orthonormal(nvec, nobs, seed=randseed)

    model = Model(eigvec, data, weights)
    model.solve_coeffs()

    if not silent:
        # print "       iter    chi2/dof     drchi_E     drchi_M   drchi_tot
        # R2            rchi2"
        print "       iter        R2             rchi2"

    for k in range(niter):
        model.solve_coeffs()
        model.solve_eigenvectors(smooth=smooth)
        if not silent:
            print 'EMPCA %2d/%2d  %15.8f %15.8f' % \
                (k + 1, niter, model.R2(), model.rchi2())
            sys.stdout.flush()

    #  One last time with latest coefficients
    model.solve_coeffs()

    if not silent:
        print "R2:", model.R2()

    return model


def classic_pca(data, nvec=None):
    """
    Perform classic SVD-based PCA of the data[obs, var].

    Returns Model object
    """
    u, s, v = N.linalg.svd(data)
    if nvec is None:
        m = Model(v, data, N.ones(data.shape))
    else:
        m = Model(v[0:nvec], data, N.ones(data.shape))
    return m


def lower_rank(data, weights=None, niter=25, nvec=5, randseed=1, silent=False):
    """
    Perform iterative lower rank matrix approximation of data[obs, var]
    using weights[obs, var].

    Generated model vectors are not orthonormal and are not
    rotated/ranked by ability to model the data, but as a set
    they are good at describing the data.

    Optional:
      - niter : maximum number of iterations to perform
      - nvec  : number of vectors to solve
      - randseed : rand num generator seed; if None, don't re-initialize

    Returns Model object
    """

    if weights is None:
        weights = N.ones(data.shape)

    nvar, nobs = data.shape
    P = _random_orthonormal(nvec, nobs, seed=randseed)
    C = N.zeros((nvar, nvec))
    ii = N.where(weights > 0)
    dof = data[ii].size - P.size - nvec * nvar

    if not silent:
        print "iter     dchi2       R2             chi2/dof"

    oldchi2 = 1e6 * dof
    for blat in range(niter):
        #  Solve for coefficients
        for i in range(nvar):
            #  Convert into form b = A x
            b = data[i]  # - b[nobs]
            A = P.T  # - A[nobs, nvec]
            w = weights[i]  # - w[nobs]
            C[i] = _solve(A, b, w)  # - x[nvec]

        #  Solve for eigenvectors
        for j in range(nobs):
            b = data[:, j]  # - b[nvar]
            A = C  # - A[nvar, nvec]
            w = weights[:, j]  # - w[nvar]
            P[:, j] = _solve(A, b, w)  # - x[nvec]

        #  Did the model improve?
        model = C.dot(P)
        delta = (data - model) * N.sqrt(weights)
        chi2 = N.sum(delta[ii]**2)
        diff = data - model
        R2 = 1.0 - N.var(diff[ii]) / N.var(data[ii])
        dchi2 = (chi2 - oldchi2) / oldchi2  # - fractional improvement in chi2
        flag = '-' if chi2 < oldchi2 else '+'
        if not silent:
            print '%3d  %9.3g  %15.8f %15.8f %s' % \
                (blat, dchi2, R2, chi2 / dof, flag)
        oldchi2 = chi2

    #  normalize vectors
    for k in range(nvec):
        P[k] /= N.linalg.norm(P[k])

    m = Model(P, data, weights)
    if not silent:
        print "R2:", m.R2()

    #  Rotate basis to maximize power in lower eigenvectors
    # -> Doesn't work; wrong rotation
    # u, s, v = N.linalg.svd(m.coeff, full_matrices=True)
    # eigvec = N.zeros(m.eigvec.shape)
    # for i in range(m.nvec):
    #     for j in range(s.shape[0]):
    #         eigvec[i] += v[i,j] * m.eigvec[j]
    #
    #     eigvec[i] /= N.linalg.norm(eigvec[i])
    #
    # m = Model(eigvec, data, weights)
    # print m.R2()

    return m


class SavitzkyGolay(object):

    """
    Utility class for performing Savitzky Golay smoothing

    Code adapted from http://public.procoders.net/sg_filter/sg_filter.py
    """

    def __init__(self, width, pol_degree=3, diff_order=0):
        self._width = width
        self._pol_degree = pol_degree
        self._diff_order = diff_order
        self._coeff = self._calc_coeff(width // 2, pol_degree, diff_order)

    def _calc_coeff(self, num_points, pol_degree, diff_order=0):
        """
        Calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        num_points   means that 2*num_points+1 values contribute to the
                     smoother.

        pol_degree   is degree of fitting polynomial

        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first
                                                 derivative of function.
                     and so on ...
        """

        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....

        x = N.arange(-num_points, num_points + 1, dtype=int)
        monom = lambda x, deg: math.pow(x, deg)

        A = N.zeros((2 * num_points + 1, pol_degree + 1), float)
        for i in range(2 * num_points + 1):
            for j in range(pol_degree + 1):
                A[i, j] = monom(x[i], j)

        # calculate diff_order-th row of inv(A^T A)
        ATA = N.dot(A.transpose(), A)
        rhs = N.zeros((pol_degree + 1,), float)
        rhs[diff_order] = (-1)**diff_order
        wvec = N.linalg.solve(ATA, rhs)

        # calculate filter-coefficients
        coeff = N.dot(A, wvec)

        return coeff

    def __call__(self, signal):
        """
        Applies Savitsky-Golay filtering
        """
        n = N.size(self._coeff - 1) / 2
        res = N.convolve(signal, self._coeff)
        return res[n:-n]


if __name__ == '__main__':

    import numpy as np
    N.random.seed(1)
    nvar = 100
    nobs = 200
    nvec = 5
    data = N.zeros(shape=(nobs, nvar))

    #  Generate data
    x = N.linspace(0, 2 * N.pi, nobs)
    for i in range(nvar):
        for k in range(nvec):
            c = N.random.normal()
            data[:, i] += 5.0 * nvec / (k + 1)**2 * c * N.sin(x * (k + 1))

    #  Add noise
    sigma = N.ones(shape=data.shape)
    # for i in sample(range(nobs), nvar / 10):
    #    sigma[i] *= 5  # temporal noise

    for i in sample(range(nvar), nvar/5):
        sigma[:, i] *= 75  # spatial noise

    # weights = 1.0 / sigma**2
    noisy_data = data + N.random.normal(scale=sigma)
    # weights = 1.0 / sigma**2

    from pandas.stats.moments import ewmstd
    weights = 1 / ewmstd(noisy_data, com=5)
    weights[np.isnan(weights)] = 0

    print "Testing empca"
    m0 = empca(noisy_data, weights, niter=20)

    print "Testing lower rank matrix approximation"
    m1 = lower_rank(noisy_data, weights, niter=20)

    print "Testing classic PCA"
    m2 = classic_pca(noisy_data, nvec=5)
    print "R2", m2.R2()

    try:
        import pylab as P
    except ImportError:
        print >> sys.stderr, "pylab not installed; not making plots"
        sys.exit(0)

    # P.subplot(111)
    # avg = np.nanmean(data, 0)
    # avg_noise = np.nanmean(noisy_data, 0)
    # ori = np.vstack((avg, avg_noise)).T
    # P.plot(ori)

    # P.figure()

    # factor score (projected X)
    nvec = 5
    if True:
        P.subplot(211)
        for i in range(nvec):
            P.plot(m0.coeff[:, i])
        # P.ylim(-0.2, 0.2)
        P.ylabel("EMPCA")
        P.title("PCA scores")

        # P.subplot(312)
        # for i in range(nvec):
        #     P.plot(m1.coeff[:, i])
        # # P.ylim(-0.2, 0.2)
        # P.ylabel("Lower Rank")

        P.subplot(212)
        for i in range(nvec):
            P.plot(m2.coeff[:, i])
        # P.ylim(-0.2, 0.2)
        P.ylabel("Classic PCA")

    P.show()
