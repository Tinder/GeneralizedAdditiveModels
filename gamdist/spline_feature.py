# Passing untrusted user input may have unintended consequences. 
# Not designed to consume input from unknown sources (i.e., 
# the public internet).

import sys
import math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
from .feature import _Feature

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef np.ndarray[DTYPE_t, ndim=1] sample(np.ndarray[DTYPE_t, ndim=1] x):
def _sample(x):
    """Sample points from an array.

    Returns a list with any duplicates or points too close together culled.

    Parameters
    ----------
    x : (n,) ndarray
        Array of floats.

    Returns
    -------
    y : (K,) ndarray
        Array of points sampled from x.

    Notes
    -----
    This function samples points from a list x. First we remove any
    duplicates. If there are fewer than 50 unique elements, we return all of
    them. Otherwise, we grow the number of sampled points logarithmically
    with the size of x. If x has n unique points, the number of points
    returned is 50 + floor(alpha * log10(n - 49)), with alpha chosen so that
    when there are 5000 unique points, 204 knots are used, consistent with
    S-PLUS, per [HTF11]. At most 1000 knots will be selected. We then return
    a sampling of so many unique points from x. The least and greatest
    elements of x are included.
    """

    #cdef long i, j, k, n, numPoints, numSamples, maxSamples
    #cdef np.ndarray[DTYPE_t, ndim=1] y, z
    #cdef double minDist, alpha, zz

    numPoints = 100
    minDist = 1e-3
    alpha = 28.2
    maxSamples = 1000

    n = len(x)
    if n <= 1:
        return x

    # Eliminate duplicate or closely spaced points
    y = np.sort(x)
    z = np.zeros(n)
    z[0] = y[0]
    j = 0
    for i in range(1,n):
        if y[i] > z[j] + minDist:
            j += 1
            z[j] = y[i]

    if j < numPoints:
        return z[0:(j+1)]

    numSamples = numPoints + int(math.floor(alpha * math.log10(j + 2 - numPoints)))
    if numSamples > maxSamples:
        numSamples = maxSamples

    if j < numSamples:
        return z[0:(j+1)]

    y = np.zeros(numSamples)
    zz = float(j+1) / numSamples
    for i in range(numSamples-1):
        k = int(math.floor(i * zz))
        y[i] = z[k]
    y[numSamples-1] = z[j]
    return y


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef np.ndarray[DTYPE_t, ndim=1] evaluate_spline_basis(double X, double[:] xi):
def _evaluate_spline_basis(X, xi):
    """Evaluate spline basis on data.

    Parameters
    ----------
    X : float
        Point at which to evaluate the spline basis.
    xi : (n,) ndarray
        Array of knots.

    Returns
    -------
    N : (K,) ndarray
        Spline basis vector.

    Notes
    -----
    This function evaluates the vector N = [N_1, N_2, ..., N_K] where
       K = len(xi),
       N_1 = 1,
       N_2 = X,
       N_{k+2} = d_k(X) - d_{K-1}(X) for 1 <= k <= K-2, and
       d_k(X) = frac{(X - xi_k)_+^3 - (X - xi_K)_+^3}{xi_K - xi_k}
    """

    ##assert xi.dtype == DTYPE

    #cdef long k, K
    #cdef np.ndarray[DTYPE_t, ndim=1] N
    #cdef double[:] d
    #cdef double term1, term2, lastXi, lastD

    K = len(xi)

    if K <= 1:
        raise SplineError('Must have at least two knots.')

    N = np.ones(K)
    N[1] = X

    d = np.zeros(K-1)
    lastXi = xi[K-1]
    for k in range(K-1):
        if X > xi[k]:
            term1 = X - xi[k]
            d[k] = term1 * term1 * term1
        else:
            continue

        if X > lastXi:
            term2 = X - lastXi
            d[k] -= term2 * term2 * term2

        d[k] /= lastXi - xi[k]

    lastD = d[K-2]
    for k in range(K-2):
        N[k+2] = d[k] - lastD

    return N

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef np.ndarray[DTYPE_t, ndim=2] omega_curvature(double[:] xi):
def _omega_curvature(xi):
    """Evaluate integrated curvature matrix.

    Parameters
    ----------
    xi : (K,) ndarray
        Array of knots.

    Returns
    -------
    Omega : (K,K)
        Integrated curvature matrix. See [HTF11].

    Notes
    -----
    This function computes the Omega matrix representing the curvature
    integral. The first two rows and columns are zeros. The remaining
    elements are given by a surprisingly complex formula:
      Omega_{ij} = 12 left( frac{xi_{K-1} - xi{j-2}}{xi_K - xi_{j-2}} right)
                       left( frac{1}{xi_K - xi_{i-2}} right)
                       left[ xi_{K-1} left( xi_K - frac{1}{2} (xi_{i-2} + xi_{j-2}) right)
                              - xi_{i-2} xi_K
                              + xi_{j-2} left( frac{3}{2} xi_{i-2} - frac{1}{2} xi_{j-2} right) right],
    where j >= i (Omega is symmetric). For the diagonal elements, where i=j,
    this simplifies considerably:
      Omega_{ii} = 12 frac{ (xi_{K-1}) - xi_{i-2})^2 }{ xi_K - xi_{i-2} }
                  = 12 * term * term / (xi_k - xi_{i-2})
    where term = xi_{K-1} - xi_{i-2}
    """

    ##assert xi.dtype == DTYPE

    #cdef long i, j, K
    #cdef np.ndarray[DTYPE_t, ndim=2] Omega
    #cdef double term, term1, term2, lastXi, secondLastXi

    K = len(xi)
    if K <= 1:
        raise SplineError('Must have at least two knots.')

    Omega = np.zeros((K, K))

    lastXi = xi[K-1]
    secondLastXi = xi[K-2]

    for i in range(2, K):
        term = secondLastXi - xi[i-2]
        Omega[i,i] = 12.0 * term * term / (lastXi - xi[i-2])

        for j in range(i+1,K):
            term1 = (secondLastXi - xi[j-2]) / (lastXi - xi[j-2])
            term1 *= 12.0 / (lastXi - xi[i-2])
            term2 = secondLastXi * (lastXi - 0.5 * (xi[i-2] + xi[j-2]))
            term2 -= xi[i-2] * lastXi
            term2 += xi[j-2] * (1.5 * xi[i-2] - 0.5 * xi[j-2])
            Omega[i,j] = term1 * term2

        for j in range(2, i):
            Omega[i,j] = Omega[j,i]
    return Omega

def _determine_smoothing(NtN, Omega, dof, lmbdaLow=0, lmbdaHigh=1, tolerance=1e-12):
    # Initial step to find upper bound on lambda
    A = NtN + lmbdaHigh * Omega

    # If all we care about is the degrees of freedom, we can get that
    # pretty quickly. The degrees of freedom is equal to the trace of the
    # smoothing matrix, S = N * (N' * N + lambda * Omega)^{-1} * N', with N
    # n-by-K. Since S is n-by-n, evaluating S takes at least O(n^2) time.
    # However, trace(A*B) = trace(B*A), so we can cyclically permute the
    # matrices to find trace(S) = trace( (N' * N + lambda * Omega)^{-1} * N' * N ).
    # We actually already evaluated N' * N (in O(n*k^2) time); we merely need to
    # solve the K equations, which takes O(K^3) time. When K << n, we win!
    ASolveNtN = la.solve(A, NtN, sym_pos=True, check_finite=False)

    while (ASolveNtN.trace() > dof):
        lmbdaLow = lmbdaHigh
        lmbdaHigh *= 2.0

        A = NtN + lmbdaHigh * Omega
        ASolveNtN = la.solve(A, NtN, sym_pos=True, check_finite=False)

    # If we made lmbda much larger, we should increase
    # tolerance proportionally.
    tolerance *= lmbdaHigh

    while (lmbdaHigh > lmbdaLow + 2.0*tolerance):
        lmbda = 0.5 * (lmbdaLow + lmbdaHigh)
        A = NtN + lmbda * Omega
        ASolveNtN = la.solve(A, NtN, sym_pos=True, check_finite=False)

        if (ASolveNtN.trace() > dof):
            lmbdaLow = lmbda
        else:
            lmbdaHigh = lmbda
    lmbda = 0.5 * (lmbdaLow + lmbdaHigh)
    return lmbda

class _SplineFeature(_Feature):
    def __init__(self, name=None, transform=None, rel_dof=4.0, load_from_file=None):
        if load_from_file is not None:
            self._load(load_from_file)
            return

        if name is None:
            raise ValueError('Feature must have a name.')

        _Feature.__init__(self, name)

        if transform is not None:
            self._has_transform = True
            self._transform = transform
        else:
            self._has_transform = False

        self._rel_dof = rel_dof

    def initialize(self, x, smoothing=1.0, save_flag=False, save_prefix=None, verbose=False):
        """Initialize variables

        Parameters
        ----------
        x : (n,) ndarray
            Features.
        """

        if self._has_transform:
            self._x = self._transform(x)
        else:
            self._x = x

        self._num_obs = len(self._x)
        self._smoothing = smoothing
        self._verbose = verbose
        self._xi = _sample(self._x)
        num_knots = len(self._xi)

        if self._verbose:
            print 'Number of observations: {0:d}'.format(self._num_obs)
            print 'Number of knots: {0:d}'.format(num_knots)

        N = np.zeros((self._num_obs, num_knots))
        for i in range(self._num_obs):
            N[i,:] = _evaluate_spline_basis(self._x[i], self._xi)

        self._N = N
        self._NtN = N.transpose().dot(N)
        self._Omega = _omega_curvature(self._xi)
        self._theta = np.zeros(num_knots)
        self._lmbda = _determine_smoothing(self._NtN, self._Omega, self._rel_dof)
        self._computed_cho_factor = False
        self._cho_factor = None

        if self._smoothing == 1.0:
            self._dof = self._rel_dof
        else:
            A = self._NtN + self._smoothing * self._lmbda * self._Omega
            self._dof = la.solve(A, self._NtN, sym_pos=True, check_finite=False).trace()

        if save_flag:
            self._save_self = True
            if save_prefix is None:
                self._filename = '{0:s}.pkcl'.format(self._name)
            else:
                self._filename = '{0:s}_{1:s}.pkcl'.format(save_prefix, self._name)
            self._save()
        else:
            self._filename = None
            self._save_self = False

    def _save(self):
        """Save parameters so model fitting can be continued later."""
        mv = {}
        mv['name'] = self._name
        mv['rel_dof'] = self._rel_dof
        mv['x'] = self._x
        mv['num_obs'] = self._num_obs
        mv['smoothing'] = self._smoothing
        mv['verbose'] = self._verbose
        mv['xi'] = np.asarray(self._xi)
        mv['N'] = self._N
        mv['NtN'] = self._NtN
        mv['Omega'] = self._Omega
        mv['theta'] = self._theta
        mv['lmbda'] = self._lmbda
        mv['computed_cho_factor'] = self._computed_cho_factor
        mv['cho_factor'] = self._cho_factor
        mv['dof'] = self._dof
        mv['save_self'] = self._save_self

        f = open(self._filename, 'w')
        pickle.dump(mv, f)
        f.close()

    def _load(self, filename):
        """Load parameters from a previous model fitting session."""
        f = open(filename)
        mv = pickle.load(f)
        f.close()

        self._filename = filename
        self._name = mv['name']
        self._rel_dof = mv['rel_dof']
        self._x = mv['x']
        self._num_obs = mv['num_obs']
        self._smoothing = mv['smoothing']
        self._verbose = mv['verbose']
        self._xi = mv['xi']
        self._N = mv['N']
        self._NtN = mv['NtN']
        self._Omega = mv['Omega']
        self._theta = mv['theta']
        self._lmbda = mv['lmbda']
        self._computed_cho_factor = mv['computed_cho_factor']
        self._cho_factor = mv['cho_factor']
        self._dof = mv['dof']
        self._save_self = mv['save_self']

    def optimize(self, fpumz, rho):
        """Optimize this Feature's parameters.

        Solves the optimization problem:
           minimize \rho/2 * \| N*\theta - y \|_2^2 + \lambda * \theta' * Omega * \theta
        with variable theta and y = N * \theta_old - fpumz where
        \theta_old is the old estimate of \theta

        Parameters
        ----------
        fpumz : (m,) ndarray
            Vector representing \bar{f}^k + u^k - \bar{z}^k
        rho : float
            ADMM parameter. Must be positive.

        Returns
        -------
        fkp1 : (m,) ndarray
           Vector representing this Feature's contribution to the response.
        -----
        """
        y = self._N.dot(self._theta) - fpumz

        Nty = self._N.transpose().dot(y)
        if not self._computed_cho_factor:
            nu = 2. * self._smoothing * self._lmbda / rho
            A = self._NtN + nu * self._Omega
            self._cho_factor = la.cho_factor(A)
            self._c = np.mean(self._N, axis=0)
            self._w = la.cho_solve(self._cho_factor, self._c)
            self._constant = 2. / (self._c.dot(self._w))
            self._computed_cho_factor = True

        self._theta = la.cho_solve(self._cho_factor, Nty, check_finite=False)
        #self._theta = la.cho_solve(self._cho_factor, Nty, check_finite=False)
        # Enforce constraint that average prediction over the data is zero.
        # Only requires an addition O(K) operations
        #self._theta -= (self._constant * (self._w.dot(Nty))) * self._w

        if self._save_self:
            self._save()


        return self._N.dot(self._theta)

    def compute_dual_tol(self, y):
        """Computes this Feature's contribution to the dual residual tolerance.

        See gamdist.
        """
        Aty = self._N.transpose().dot(y)
        return Aty.dot(Aty)

    def num_params(self):
        """Number of parameters.

        Returns the number of parameters used in this component of the
        model. This is for a different purpose than the degrees of
        freedom. The latter is used for model selection; the former for
        assessing convergence. Thus, the presence of regularization does
        not impact this function.
        """
        return len(self._theta)

    def dof(self):
        """Degrees of freedom.

        Returns the degrees of freedom associated with this component of
        the model. Spline features nominally have a degree of freedom
        for each knot point; however, the use of regularization can
        dramatically reduce the effective degrees of freedom.

        Returns
         dof : float
             Degrees of freedom.
        """
        return self._dof

    def predict(self, X):
        """Apply fitted model to feature.

        Parameters
        ----------
         X : array
             Data for this feature. If a transformation was
             specified for this feature, the provided data
             should be "pre-transformation".

        Returns
        -------
         f_j : array
             The contribution of this feature to the predicted
             response.
        """
        if self._has_transform:
            xx = self._transform(X)
        else:
            xx = X

        num_obs = len(xx)
        num_knots = len(self._xi)
        N = np.zeros((num_obs, num_knots))

        for i in range(num_obs):
            N[i,:] = _evaluate_spline_basis(xx[i], self._xi)
        return N.dot(self._theta)

    def _plot(self, true_fn=None):
        ''' Creates a graph showing the transformed response as a function of the feature. '''
        # Plot distribution of training examples as a histogram at the bottom.

        num_obs = 100
        x_plot = np.linspace(np.min(self._x), np.max(self._x), num=num_obs)

        num_knots = len(self._xi)
        N = np.zeros((num_obs, num_knots))
        for i in range(num_obs):
            N[i,:] = _evaluate_spline_basis(x_plot[i], self._xi)

        y_hat = N.dot(self._theta)

        # If we're plotting, we need to compute the whole smoothing matrix.
        # Form S matrix to compute degrees of freedom
        #S = N.dot(la.solve(A, N.transpose(), sym_pos=True, check_finite=False))
        # ^ No need to store an n-by-n matrix; we only need to store a k-by-n matrix
        A = N.transpose().dot(N) + (self._smoothing * self._lmbda) * self._Omega
        S = la.solve(A, N.transpose(), sym_pos=True, check_finite=False)
        se = np.zeros(num_obs)
        for i in range(num_obs):
            se[i] = la.norm(N[i,:].dot(S))
        ub = y_hat + 2 * se
        lb = y_hat - 2 * se

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.fill_between(x_plot, lb, ub, color='y')
        ax.plot(x_plot, y_hat, 'g-')
        if true_fn is not None:
            y_true = true_fn(x_plot)
            ax.plot(x_plot, y_true, 'k--')
        plt.xlabel(self._name, fontsize=24)
        plt.ylabel(r'$f_\textrm{' + self._name + '}$', fontsize=24)
        title = r'$\textrm{df}_\lambda = '
        dollarSign = r'$'
        plt.title('{0:s}{1:.0f}{2:s}'.format(title, self._dof, dollarSign), fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.show()



    def __str__(self):
        ''' Print confidence intervals on level values, significance tests? '''
        # For spline features, print the degrees of freedom,
        # and a p-value against the null hypothesis that
        # the spline is 0.
        return 'Feature {0:s} (spline): {1:0.0f} dof\n'.format(self._name, self._dof)
