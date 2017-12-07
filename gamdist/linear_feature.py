# Passing untrusted user input may have unintended consequences. 
# Not designed to consume input from unknown sources (i.e., 
# the public internet).

import numpy as np
from .feature import _Feature

class _LinearFeature(_Feature):
    def __init__(self, name=None, transform=None, regularization=None, load_from_file=None):
        """Initialize feature model (independent of data).

        This function is called when the user adds a feature to a
        Model, before any data has been specified.

        Parameters
        ----------
        name : string
           Name for feature, used to make plots.
        transform : function
           Transformation applied to the data.
        regularization : dictionary
            Description of regularization terms. Two kinds of
            regularization are supported: l1 and l2. To include one or
            more of these types of regularization, include a key/value
            specifying the details in this dictionary. The keys should
            be in ['l1', 'l2']. The corresponding value provides
            additional details on the regularization, as described
            below. One additional key is recognized in this dictionary,
            'prior', specifying a prior estimate of the parameters. See
            below for more details.
        load_from_file : string or None.
            Filename for this Feature, to be used for loading parameters. If
            None (default), parameters are not loaded. If this parameter
            is specified, any other parameters are ignored.

        l1 Regularization Parameters
        ----------------------------
        Description
            l1 regularization is said to encourage sparsity, in the sense
            that the resulting parameter estimate will typically match
            the prior estimates perfectly. On the other hand, it does not
            heavily discourage large deviations.
        coef : float
            Coefficient associated with this regularization term.
        prior : float
            Prior estimate of the parameter for this feature. If this
            variable is not specified at all, an estimate of zero is
            assumed. Note: this variable applies to both l1 and l2
            regularization, so if both are included, it seems silly
            to specify it twice. Therefore, and perhaps confusingly, this
            variable is specified in the top level regularization dictionary,
            not in the individual regularization term dictionary. This is
            best illustrated by example:
               regularization = {'l1': {'coef': 0.3},
                                 'l2': {'coef': 0.3},
                                 'prior': 1.7
                                 }
            This incorporates both types of regularization, with a coefficient
            of 0.3 applied to all terms and a prior estimate of 1.7.

        l2 Regularization Parameters
        ----------------------------
        Description
            l2 regularization discourages large deviations from the prior
            estimate.
        coef : float
            See l1 Regularization Parameters.
        prior: float
            See l1 Regularization Parameters.

        """

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

        self._has_l1 = False
        self._has_l2 = False
        if regularization is not None:
            if 'l1' in regularization:
                self._has_l1 = True
                if 'coef' in regularization['l1']:
                    self._coef1 = regularization['l1']['coef']
                else:
                    raise ValueError('No coefficient specified for l1 regularization term.')

            if 'l2' in regularization:
                self._has_l2 = True
                if 'coef' in regularization['l2']:
                    self._coef2 = regularization['l2']['coef']
                else:
                    raise ValueError('No coefficient specified for l2 regularization term.')

            if (self._has_l1 or self._has_l2):
                self._has_prior = True
                self._prior = regularization.get('prior', 0.0)
            else:
                self._has_prior = False


    def initialize(self, x, smoothing=1.0, save_flag=False, save_prefix=None, verbose=False):
        """Initialize data associated with this feature.

        Completes the initialization of this feature based on the data
        used to fit the model.

        Parameters
        ----------
         x : array
             Observations corresponding to this feature.
         smoothing : float
             Multiplicative factor by which we modify any regularization
             associated with this model. Defaults to 1.0.
         save_flag : boolean
             Indicates whether to save results after each iteration.
             Defaults to False.
         save_prefix : str or None
             If save_flag is True, the file in which these data are
             saved will have a prefix specified by this option. If no
             prefix is specified, the file will simply reflect the
             name of this feature.
         verbose : boolean
             Specifies whether to print mildly helpful information.
             Defaults to False.

        Returns
        -------
         (nothing)
        """
        if self._has_transform:
            xx = self._transform(x)
            self._xmean = np.mean(xx)
            self._x = xx - self._xmean
        else:
            self._xmean = np.mean(x)
            self._x = x - self._xmean

        self._xtx = self._x.dot(self._x)
        self._m = 0.0
        self._b = 0.0

        if self._has_l1:
            self._lambda1 = self._coef1 * smoothing

        if self._has_l2:
            self._lambda2 = self._coef2 * smoothing

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
        mv['xmean'] = self._xmean
        mv['x'] = self._x
        mv['xtx'] = self._xtx
        mv['m'] = self._m
        mv['b'] = self._b
        mv['has_l1'] = self._has_l1
        if self._has_l1:
            mv['lambda1'] = self._lambda1
        mv['has_l2'] = self._has_l2
        if self._has_l2:
            mv['lambda2'] = self._lambda2
        mv['has_prior'] = self._has_prior
        if self._has_prior:
            mv['prior'] = self._prior
        mv['verbose'] = self._verbose
        mv['name'] = self._name
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
        self._xmean = mv['xmean']
        self._x = mv['x']
        self._xtx = mv['xtx']
        self._m = mv['m']
        self._b = mv['b']
        self._has_l1 = mv['has_l1']
        if self._has_l1:
            self._lambda1 = mv['lambda1']
        self._has_l2 = mv['has_l2']
        if self._has_l2:
            self._lambda2 = mv['lambda2']
        self._has_prior = mv['has_prior']
        if self._has_prior:
            self._prior = mv['prior']
        self._verbose = mv['verbose']
        self._name = mv['name']
        self._save_self = mv['save_self']

    def optimize(self, fpumz, rho):
        """Optimize this Feature's parameters.

        Solves the optimization problem:
           minimize \rho/2 * \| y - (m*x + b) \|_2^2
                      + \lambda1 * | b - b_prior |
                      + \lambda2 * (b - b_prior)^2
           s.t.     m*xbar + b = 0
        with variables m, b. In this case, y = m_prior * x_j + b_prior - fpumz

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

        Notes
        -----
        The unconstrained problem listed above has a non-smooth
        objective. It is equivalent to the following quadratic
        program with linear inequality constraints:
           minimize



           minimize q' * (A'*A + \nu_2 * I) * q + 2 * q' * A'*b
                        + \nu_1 * sum(t)
                        + \nu_nl * sum(s)
           subject to   0 <= t
                        0 <= s
                        -t <= q <= t
                        -s <= D*q <= s
        with optimization variables q, t, s. We use cvxpy to solve this
        equivalent problem. The size of this problem is independent of
        the number of training examples, and is governed strictly by the
        number of categories, and the specified number of connections.
        In our tests, cvxpy performs admirably. In situations where we
        have a large number of categories, or the number of connections
        is large, it might make sense to invest the time to write a
        custom solver that takes into account the structure of the
        problem. Specifically, noting that A'*A is a diagonal matrix,
        and D has a specific sparsity pattern: each row of D contains
        two non-zero elements, one equal to +1, the other equal to -1.
        """

        y = self._m * self._x - fpumz

        if self._has_l2:
            denom = self._xtx + 2 * self._lambda2 / rho
        else:
            denom = self._xtx

        self._m = self._x.dot(y) / denom
        self._b = - self._m * self._xmean

        if self._save_self:
            self._save()

        return self._m * self._x

    def compute_dual_tol(self, y):
        """Compute this Feature's contribution to the dual residual tolerance.

        See gamdist.
        """
        ybar = np.sum(y)
        xty = self._x.dot(y)
        return (xty + 2 * self._xmean * ybar) * xty  + (1. + self._xmean * self._xmean) * ybar * ybar

    def num_params(self):
        """Number of parameters.

        Returns the number of parameters used in this component of the
        model. This is for a different purpose than the degrees of
        freedom. The latter is used for model selection; the former for
        assessing convergence. Thus, the presence of regularization does
        not impact this function.
        """
        return 1

    def dof(self):
        """Degrees of freedom.

        Returns the degrees of freedom associated with this component of
        the model. Linear features nominally have 2 degrees of freedom
        corresponding to the slope and intercept. We constrain the
        intercept since we include an affine term in the overall
        model. That reduces the degrees of freedom by one. When using
        regularization, the effective degrees of freedom are even lower,
        but I haven't researched this part yet. Thus we just return 1.

        Returns
         dof : float
             Degrees of freedom.
        """
        return 1

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
            return self._m * self._transform(X) + self._b
        else:
            return self._m * X + self._b


    def _plot(self):
        ''' Creates a graph showing the transformed response as a function of the feature. '''
        # For linear features, plot a line showing the slope with
        # confidence interval and distribution of observations.
        # If Binomial, show 0/1 observations on top/bottom.
        pass

    def __str__(self):
        ''' Print confidence intervals on slope, significance tests? '''
        # For linear features, print slope, a confidence
        # interval, and a p-value against the null hypothesis
        # that the slope is 0.
        return 'Feature {0:s}: beta = {1:.06g}\n'.format(self._name, self._m)
