# Passing untrusted user input may have unintended consequences. 
# Not designed to consume input from unknown sources (i.e., 
# the public internet).

import sys
import numpy as np
import cvxpy as cvx
from scipy import sparse
from .feature import _Feature
import pickle

class _CategoricalFeature(_Feature):
    def __init__(self, name=None, regularization=None, load_from_file=None):
        """Initialize feature model (independent of data).

        This function is called when the user adds a feature to a
        Model, before any data has been specified.

        Parameters
        ----------
        name : string
            Name for feature, used to make plots.
        regularization : dictionary
            Description of regularization terms. Three kinds of
            regularization are supported: l1, l2, and Network Lasso.
            To include one or more of these types of regularization,
            include a key/value specifying the details in this dictionary.
            The keys should be in ['l1', 'l2', 'network_lasso']. The
            corresponding value provides additional details on the
            regularization, as described below. One additional key
            is recognized in this dictionary, 'prior', specifying
            a prior estimate of the parameters. See below for more
            details.
        load_from_file : string or None.
            Filename for this Feature, to be used for loading parameters. If
            None (default), parameters are not loaded. If this parameter
            is specified, any other parameters are ignored.

        l1 Regularization Parameters
        ----------------------------
        Description
            l1 regularization is said to encourage sparsity, in the sense
            that the resulting parameter estimates will typically match
            some of the prior estimates perfectly. On the other hand,
            it does not heavily discourage large deviations.
        coef : float or dictionary
            Coefficient associated with this regularization term, applied
            to all parameter estimates (if float), or with a separate
            coefficient for multiple parameters (if dictionary). If a
            dictionary is used, only parameters with specified coefficients
            are regularized.
        prior : dictionary
            Prior estimate of parameters associated with different categories.
            Only parameters with specified prior estimates are regularized.
            If this variable is not specified at all, an estimate of zero
            for all categories is assumed. Note: this variable applies to both
            l1 and l2 regularization, so if both are included, it seems silly
            to specify it twice. Therefore, and perhaps confusingly, this
            variable is specified in the top level regularization dictionary,
            not in the individual regularization term dictionary. This is
            best illustrated by example:
               regularization = {'l1': {'coef': 0.3},
                                 'l2': {'coef': {'male': 0.1, 'female': 0.2}},
                                 'network_lasso': {'coef': 0.3, 'edges': edges},
                                 'prior': {'male': -3.0, 'female': 5.0}
                                 }
            This incorporates all three types of regularization, with a coefficient
            of 0.3 applied to all terms in the l1 term, a coefficient of 0.1 (0.2)
            applied to the male (female) components of the l2 term, a prior estimate
            of -3.0 (5.0) for the parameters corresponding to males (females), which
            is applicable to both the l1 and l2 norms, and finally, a Network Lasso
            term with overall coefficient 0.3, and edges specified in the pandas
            DataFrame.

        l2 Regularization Parameters
        ----------------------------
        Description
            l2 regularization discourages large deviations from the prior
            estimate.
        coef : float or dictionary
            See l1 Regularization Parameters.
        prior: dictionary
            See l1 Regularization Parameters.

        Network Lasso Regularization Parameters
        ---------------------------------------
        Description
            The Network Lasso encourages similar categories to have similar
            parameter values, where the similarity of categories is defined
            by the user.
        coef : float
            See l1 Regularization Parameters; but! Only floats are permitted
            here.
        edges : pandas DataFrame
            Specifies which categories are believed to be similar to other
            categories. Dataframe must have at least two columns called 'country1'
            and 'country2'. An optional third column called 'weight' specifies the
            strength of the belief. All values will be scaled to have maximum value
            equal to 1 (so only relative strengths are used). Use coef to capture
            the overall strength of these beliefs. If the 'weight' column is not
            present, weights of 1 will be used for all parameters.

        Optional Parameters
        -------------------
        use_cvx : bool
            Flag specifying whether to use CVXPY to solve the
            optimization problem. Defaults to True, and since no other
            method has been implemented, does nothing.
        solver : string
            Solver to use with CVXPY. Defaults to 'ECOS'.
        """

        if load_from_file is not None:
            self._load(load_from_file)
            return

        if name is None:
            raise ValueError('Feature must have a name.')

        _Feature.__init__(self, name)

        self._use_cvx = True
        self._solver = 'ECOS'
        self._categories = []
        self._has_l1 = False
        self._has_l2 = False
        self._has_network_lasso = False
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

            if 'network_lasso' in regularization:
                self._has_network_lasso = True
                if 'coef' in regularization['network_lasso']:
                    self._lambda_network_lasso = regularization['network_lasso']['coef']
                else:
                    raise ValueError('No coefficient specified for Network Lasso regularization term.')

                if 'edges' in regularization['network_lasso']:
                    self._edges = regularization['network_lasso']['edges']
                    self._num_edges, em = self._edges.shape
                    # Loop over edges to see if a node is listed there that isn't
                    # present in the data. This matters!  That means a category exists;
                    # we simply have no hard data about the probability associated with
                    # that category. We will *still* be able to predict a probability
                    # by the graph structure, provided it is connected to at least one
                    # node for which we *do* have hard data.
                    for index, row in self._edges.iterrows():
                        if row['country1'] not in self._categories:
                            self._categories.append(row['country1'])
                        if row['country2'] not in self._categories:
                            self._categories.append(row['country2'])

                    # Wait to create D until we have the data
                else:
                    raise ValueError('Edges not specified for Network Lasso regularization term.')

            if (self._has_l1 or self._has_l2) and 'prior' in regularization:
                self._has_prior = True
                self._prior = regularization['prior']
            else:
                self._has_prior = False


    def initialize(self, x, smoothing=1.0, save_flag=False, save_prefix=None, na_signifier=None, verbose=False):
        """Initialize variables once data has been specified.

        Parameters
        ----------
        x : array
            Observation corresponding to this feature.
        smoothing : float
            Overall smoothing factor, on top of the relative
            smoothing specified in __init__. Defaults to 1.0,
            indicating no change to the default smoothing.
        save_flag : boolean
            Flag indicating whether to save intermediate results.
        save_prefix : string or None.
            Prefix for filename. Actual filename will use this prefix
            followed by the name of this feature, with a .pckl extension.
        na_signifier : string or None
            Indicating how missing data is marked. If specified, missing
            data is handled by a special category whose parameter is
            fixed at 0. Defaults to None.
        verbose : bool
            Flag specifying whether to print mildly helpful info.
            Defaults to False.

        """
        self._num_obs = len(x)
        self._verbose = verbose

        # Final list of categories consists of all categories specified via
        # regularization terms as well as all the categories listed in the data
        self._categories = list(set(x).union(self._categories))
        self._num_categories = len(self._categories)
        self._category_hash = {key: i for (key, i) in zip(self._categories, range(self._num_categories))}

        # If we are using the Network Lasso, compute the relevant matrix, D.
        if self._has_network_lasso:
            D = np.zeros((self._num_edges, self._num_categories))
            ne, em = self._edges.shape
            ir = 0
            for index, row in self._edges.iterrows():
                # Check that nodes are in data
                i = self._category_hash[row['country1']]
                j = self._category_hash[row['country2']]
                if em >= 3:
                    lmbda = row['weight']
                else:
                    lmbda = 1.

                D[ir, i] = lmbda
                D[ir, j] = -lmbda
                ir += 1
            self._D = sparse.coo_matrix(D).tocsr()
            self._lambda_network_lasso *= smoothing

        # If there is a value corresponding to "unknown category", label it
        # appropriately. We will make sure the corresponding parameter is
        # always 0, since we make no prediction in this case.
        if na_signifier is not None and na_signifier in self._categories:
            self._na_index = self._category_hash[na_signifier]
        else:
            self._na_index = -1

        # Store, not the data themselves, but integers representing the
        # categories.
        self.x = np.zeros(self._num_obs, dtype=np.int)
        cnt = np.zeros(self._num_categories, dtype=np.int)
        for (ix, i) in zip(x, range(self._num_obs)):
            cnt[self._category_hash[ix]] += 1
            self.x[i] = self._category_hash[ix]

        # Replace coef1 (a float or dictionary) with lambda1 (a vector of weights)
        # Possibility #1: coef1 is a float, no prior provided. In this case,
        # all categories get the same weight, coef1 * smoothing.
        #
        # Possibility #2: coef1 is a float, prior provided. In this case,
        # lambda should put weight zero on any categories not represented in the prior.
        # all categories with prior specified get the same weight, coef1 * smoothing.
        #
        # Possibility #3: coef1 is a dictionary, no prior. In this case,
        # lambda should put weight zero on any categories not represented in coef1.
        # Any categories represented in coef1 get weight coef1[category] * smoothing.
        #
        # Possibility #4: coef1 is a dictionary, prior provided. In this case,
        # lambda should put weight zero on any categories not represented *in both*
        # coef1 and prior. Any categories represented in both coef1 and prior
        # get weight coef1[category] * smoothing
        if self._has_l1:
            if type(self._coef1) == float:
                if self._has_prior:
                    self._lambda1 = np.zeros(self._num_categories)
                    l = self._coef1 * smoothing
                    for key in self._prior:
                        self._lambda1[self._category_hash[key]] = l
                else:
                    self._lambda1 = np.full(self._num_categories, self._coef1 * smoothing)
            else:
                if self._has_prior:
                    self._lambda1 = np.zeros(self._num_categories)
                    for key, value in self._coef1.iteritems():
                        if key in self._prior:
                            self._lambda1[self._category_hash[key]] = value * smoothing
                else:
                    self._lambda1 = np.zeros(self._num_categories)
                    for key, value in self._coef1.iteritems():
                        self._lambda1[self._category_hash[key]] = value * smoothing


        if self._has_l2:
            if type(self._coef2) == float:
                if self._has_prior:
                    self._lambda2 = np.zeros(self._num_categories)
                    l = self._coef2 * smoothing
                    for key in self._prior:
                        self._lambda2[self._category_hash[key]] = l
                else:
                    self._lambda2 = np.full(self._num_categories, self._coef2 * smoothing)
            else:
                if self._has_prior:
                    self._lambda2 = np.zeros(self._num_categories)
                    for key, value in self._coef2.iteritems():
                        if key in self._prior:
                            self._lambda2[self._category_hash[key]] = value * smoothing
                else:
                    self._lambda2 = np.zeros(self._num_categories)
                    for key, value in self._coef2.iteritems():
                        self._lambda2[self._category_hash[key]] = value * smoothing

        if (self._has_l1 or self._has_l2) and self._has_prior:
            prior = np.zeros(self._num_categories)
            for key, value in self._prior.iteritems():
                prior[self._category_hash[key]] = value
            self._prior = prior


        self._AtA = sparse.dia_matrix((cnt, 0), shape=(self._num_categories, self._num_categories), dtype=np.int)
        self.p = np.zeros(self._num_categories)

        if self._verbose:
            print 'Number of categories: {0:d}'.format(self._num_categories)
            if self._has_edges:
                print 'Number of edges: {0:d}'.format(self._num_edges)

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
        mv['num_obs'] = self._num_obs
        mv['categories'] = self._categories
        mv['num_categories'] = self._num_categories
        mv['category_hash'] = self._category_hash
        mv['has_l1'] = self._has_l1
        if self._has_l1:
            mv['lambda1'] = self._lambda1
        mv['has_l2'] = self._has_l2
        if self._has_l2:
            mv['lambda2'] = self._lambda2
        mv['has_network_lasso'] = self._has_network_lasso
        if self._has_network_lasso:
            mv['num_edges'] = self._num_edges
            mv['D'] = self._D
            mv['lambda_network_lasso'] = self._lambda_network_lasso
        mv['has_prior'] = self._has_prior
        if self._has_prior:
            mv['prior'] = self._prior
        mv['na_index'] = self._na_index
        mv['x'] = self.x
        mv['p'] = self.p
        mv['AtA'] = self._AtA
        mv['verbose'] = self._verbose
        mv['use_cvx'] = self._use_cvx
        mv['solver'] = self._solver
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
        self._num_obs = mv['num_obs']
        self._categories = mv['categories']
        self._num_categories = mv['num_categories']
        self._category_hash = mv['category_hash']
        self._has_l1 = mv['has_l1']
        if self._has_l1:
            self._lambda1 = mv['lambda1']
        self._has_l2 = mv['has_l2']
        if self._has_l2:
            self._lambda2 = mv['lambda2']
        self._has_network_lasso = mv['has_network_lasso']
        if self._has_network_lasso:
            self._num_edges = mv['num_edges']
            self._D = mv['D']
            self._lambda_network_lasso = mv['lambda_network_lasso']
        self._has_prior = mv['has_prior']
        if self._has_prior:
            self._prior = mv['prior']
        self._na_index = mv['na_index']
        self.x = mv['x']
        self.p = mv['p']
        self._AtA = mv['AtA']
        self._verbose = mv['verbose']
        self._use_cvx = mv['use_cvx']
        self._solver = mv['solver']
        self._name = mv['name']
        self._save_self = mv['save_self']

    def optimize(self, fpumz, rho):
        """Optimize this Feature's parameters.

        Solves the optimization problem:
           minimize \rho/2 * \| A*q + b \|_2^2
                      + \| diag(\lambda_1) * (q - q_prior) \|_1
                      + \| diag(\lambda_2)^(1/2) * (q - q_prior) \|_2^2
                      + \lambda_nl * \| D*q \|_1
           s.t. 1' * A * q = 0
        where b = \bar{f}^k + u^k - \bar{z}^k - A*q^k

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
           minimize q' * (A'*A + (2 / \rho) * diag(\lambda_2)) * q
                        + 2 * q' * (A' * b - (2 / \rho) * diag(\lambda_2) * q_prior)
                        + (2 / \rho) * \lambda_1' * t
                        + (2 / \rho) * \lambda_nl * sum(s)
           subject to   c' * q = 0
                        0 <= t
                        0 <= s
                        -t <= q - q_prior <= t
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
        b = fpumz - self._compute_Az(self.p)
        Atb = self._compute_Atz(b)

        if self._has_l2 and self._has_prior:
            Atb -= (2. / rho) * np.multiply(self._lambda2, self._prior)

        q = cvx.Variable(self._num_categories)

        if self._has_l2:
            AtA = cvx.Constant(self._AtA + sparse.dia_matrix( ((2. / rho) * self._lambda2, 0), shape=(self._num_categories, self._num_categories), dtype=np.float))
        else:
            AtA = cvx.Constant(self._AtA)

        Atb = cvx.Constant(Atb)

        obj = cvx.quad_form(q, AtA) + 2. * (Atb.T * q)

        c = cvx.Constant(self._AtA.diagonal())
        constraints = [c.T * q == 0]

        if self._has_l1:
            t = cvx.Variable(self._num_categories)
            lmbda1 = cvx.Constant((2 / rho) * self._lambda1)
            obj += (self._lambda1.T * t)
            constraints += [0 <= t[:],
                            0 <= t[:] + q[:] - q_prior[:],
                            0 <= t[:] - q[:] + q_prior[:]]

        if self._has_network_lasso:
            s = cvx.Variable(self._num_edges)
            D = cvx.Constant(self._D)
            obj += (2 / rho) * self._lambda_network_lasso * cvx.sum_entries(s)
            constraints += [0 <= s[:],
                            0 <= s[:] + D * q[:],
                            0 <= s[:] - D * q[:]]

        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        prob.solve(verbose=self._verbose, solver=self._solver)#, abstol=1e-3, reltol=1e-3, feastol=1e-3)

        if prob.status != cvx.OPTIMAL and prob.status != cvx.OPTIMAL_INACCURATE:
            print "Categorical variable failed to converge."
            sys.exit()

        self.p = q.value.A.squeeze()
        if self._na_index >= 0:
            self.p[self.naIndex] = 0

        if self._save_self:
            self._save()

        return self._compute_Az(self.p)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #def _compute_Az(self, np.ndarray[DTYPE_t, ndim=1] z):
    def _compute_Az(self, z):
        """Compute A*z.

        The sparsity pattern of A enables us to compute A*z in time
        proportional to the number of elements of z, regardless of the
        number of rows of A.
        """
        #cdef long i, ix
        #cdef long m = self._num_obs
        #cdef long[:] x = self.x
        #cdef np.ndarray[DTYPE_t, ndim=1] Az = np.zeros(m)

        m = self._num_obs
        x = self.x
        Az = np.zeros(m)

        for i in range(m):
            ix = x[i]
            Az[i] = z[ix]
        return Az

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #def _compute_Atz(self, np.ndarray[DTYPE_t, ndim=1] z):
    def _compute_Atz(self, z):
        """Compute A' * z.

        The sparsity pattern of A enables us to compute A' * z in time
        proportional to the number of elements of z, regardless of the
        number of columns of A.
        """
        #cdef long i, ix
        #cdef long m = self._num_obs
        #cdef long K = self._num_categories
        #cdef long[:] x = self.x
        #cdef np.ndarray[DTYPE_t, ndim=1] Atz = np.zeros(K)

        m = self._num_obs
        K = self._num_categories
        x = self.x
        Atz = np.zeros(K)

        for i in range(m):
            ix = x[i]
            Atz[ix] += z[i]
        return Atz

    def compute_dual_tol(self, y):
        """Computes this Feature's contribution to the dual residual tolerance.

        See gamdist.
        """
        Aty = self._compute_Atz(y)
        return Aty.dot(Aty)

    def num_params(self):
        """Number of parameters.

        Returns the number of parameters used in this component of the
        model. This is for a different purpose than the degrees of
        freedom. The latter is used for model selection; the former for
        assessing convergence. Thus, the presence of regularization does
        not impact this function.
        """
        return len(self.p)

    def dof(self):
        """Degrees of freedom.

        Returns the degrees of freedom associated with this component of
        the model. Categorical features nominally have 1 degree of
        freedom for each category (known as "levels" among
        statisticians). Since we include an affine term in the overall
        model, there is an ambiguity in the parameters that can be
        resolved in various ways. We apply the constraint that the
        average contribution, over all observations in the training set,
        is zero. This has the effect of reducing the DOF by 1. Any
        regularization included reduces the effective DOF even further,
        but I have not had a chance to research this properly. Thus we
        just return the number of levels minus 1.

        Returns
         dof : float
             Degrees of freedom.
        """
        return len(self.p) - 1

    def predict(self, x):
        """Apply fitted model to feature.

        Parameters
        ----------
         X : array
             Data for this feature.

        Returns
        -------
         f_j : array
             The contribution of this feature to the predicted
             response.
        """
        prediction = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] in self._category_hash:
                prediction[i] = self.p[self._category_hash[x[i]]]
        return prediction


    def plot(self):
        ''' Creates a bar chart showing the transformed response for each level. '''
        # For categorical features, plot a bar graph for each category
        # showing the estimated values, confidence intervals, and number
        # of observations in each category. If Family is Binomial,
        # show 0/1 observations on top/bottom of graph. Otherwise
        # show observations on bottom of graph.
        #
        # We could permit fancier plots for geographic data e.g.
        # Choropleth maps.
        pass

    def __str__(self):
        ''' Print confidence intervals on level values, significance tests? '''
        # For categorical features, print value associated
        # with each category, a confidence interval, and
        # a p-value against the null hypothesis that the
        # parameter is 0. For Network Lasso, print p-value
        # against null hypothesis that related categories
        # have the same value.
        desc = 'Feature {0:s}\n'.format(self._name)
        for i in self._categories:
            desc += '  {0:s}: {1:.06g}\n'.format(i, self.p[self._category_hash[i]])
        return desc
