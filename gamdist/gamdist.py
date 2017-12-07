# Passing untrusted user input may have unintended consequences.
# Not designed to consume input from unknown sources (i.e., 
# the public internet).

import sys
import pickle
import multiprocessing as mp
import numpy as np
import scipy.special as special
import scipy.stats as stats
import scipy.linalg as linalg
from matplotlib import pyplot as plt
from .feature import _Feature
from .categorical_feature import _CategoricalFeature
from .linear_feature import _LinearFeature
from .spline_feature import _SplineFeature
import proximal_operators as po

# To do:
# - Save and load properly
# - Implement Multinomial, Proportional Hazards
# - Implement Binomial models for covariate classes
# - Implement outlier detection
# - Implement overdispersion for Binomial and Poisson families
# - Hierarchical models
# - AICc, BIC, R-squared estimate
# - Confidence intervals on mu, predictions (probably need to use Bootstrap but can do so intelligently)
# - Confidence intervals on model parameters, p-values
# - Constrain spline to have mean prediction 0 over the data
# - Group lasso penalty (l2 norm -- not squared -- or l_\infty norm on f_j(x_j; p_j))
# - Piecewise constant fits, total variation regularization
# - Monotone constraint
# - Interactions
# - Runtime optimization (Cython)
# - Fit in parallel
# - Residuals
#   - Compute different types of residuals (Sec 3.1.7 of [GAMr])
#   - Plot residuals against mean response, variance, predictor, unused predictor
#   - QQ plot of residuals
#
# Done:
# - Implement Gaussian, Binomial, Poisson, Gamma, Inv Gaussian,
# - Plot splines
# - Deviance (on training set and test set), AIC, Dispersion, GCV, UBRE
# - Write documentation
# - Check implementation of Gamma dispersion
# - Implement probit, complementary log-log links.

FAMILIES = ['normal',
            'binomial',
            'poisson',
            'gamma',
            'exponential',
            'inverse_gaussian'
            ]

LINKS = ['identity',
         'logistic',
         'probit',
         'complementary_log_log',
         'log',
         'reciprocal',
         'reciprocal_squared'
         ]

FAMILIES_WITH_KNOWN_DISPERSIONS = {'binomial': 1,
                                   'poisson': 1
                                   }

CANONICAL_LINKS = {'normal': 'identity',
                   'binomial': 'logistic',
                   'poisson': 'log',
                   'gamma': 'reciprocal',
                   'inverse_gaussian': 'reciprocal_squared'
                   }
# Non-canonical but common link/family combinations include:
# Binomial: probit and complementary log-log
# Gamma: identity and log

def _plot_convergence(prim_res, prim_tol, dual_res, dual_tol, dev):
    """Plot convergence progress.

    We deem the algorithm to have converged when the prime and dual
    residuals are smaller than tolerances which are themselves computed
    based on the data as in [ADMM]. Some analysts prefer to claim
    convergence when changes to the deviance (a measure of goodness of
    fit). Thus we plot that as well. Specifically, we plot, on a log
    scale, dev - dev_final, where dev_final is the deviance of the final
    model. We add 1e-10 just to avoid taking the logarithm of zero, which
    is completely arbitrary but makes the plot look acceptable.

    Parameters
    ----------
     prim_res : array
         Array of prime residuals after each iteration.
     prim_tol : array
         Array of prime tolerances after each iteration.
     dual_res : array
         Array of dual residuals after each iteration.
     dual_tol : array
         Array of dual tolerances after each iteration.
     dev : array
         Array of deviances after each iteration

    Returns
    -------
     (nothing)
    """
    fig = plt.figure(figsize=(12., 10.))

    ax = fig.add_subplot(211)
    ax.plot(range(len(prim_res)), prim_res, 'b-', label='Primal Residual')
    ax.plot(range(len(prim_tol)), prim_tol, 'b--', label='Primal Tolerance')
    ax.plot(range(len(dual_res)), dual_res, 'r-', label='Dual Residual')
    ax.plot(range(len(dual_tol)), dual_tol, 'r--', label='Dual Tolerance')
    ax.set_yscale('log')
    plt.xlabel('Iteration', fontsize=24)
    plt.ylabel('Residual', fontsize=24)
    plt.legend(fontsize=24, loc=3)

    ax = fig.add_subplot(212)
    ax.plot(range(len(dev)), (dev - dev[-1]) + 1e-10, 'b-', label='Deviance')
    ax.set_yscale('log')
    plt.xlabel('Iteration', fontsize=24)
    plt.ylabel('Deviance Suboptimality', fontsize=24)

    plt.gcf().subplots_adjust(bottom=0.1)
    plt.gcf().subplots_adjust(left=0.1)
    plt.show()

def _feature_wrapper(f):
    """Wrapper for feature optimization.

    This is a wrapper for use with multi-threaded versions.
    Unfortunately Python threads are *terrible*, so this doesn't
    actually get used.

    Parameters
    ------
     f : list
         Array of inputs. f[0] is the name of the feature. f[1]
         is the feature object itself. f[2] is N * fpumz (the
         vector input to the feature during optimization). f[3]
         is the ADMM parameter, rho.

    Returns
    -------
     name : str
         The name of the feature. (The same as the input.)
     f_j : array
         The array of fitted values returned by the feature.
    """

    return f[0], f[1].optimize(f[2], f[3])


def _gamma_dispersion(dof, dev, num_obs):
    """Gamma dispersion.

    This function estimates the dispersion of a Gamma family with p
    degrees of freedom and deviance D, and n observations. The
    dispersion nu is that number satisfying
      2*n * (log nu - psi(nu)) - p / nu = D

    We use Newton's method with a learning rate to solve this nonlinear
    equation.

    Parameters
    ----------
     dof : float
         Degrees of freedom
     dev : float
         Deviance
     num_obs : int
         Number of observations

    Returns
    -------
     nu : float
         Estimated dispersion
    """
    beta = 0.1
    tol = 1e-6
    max_its = 100

    nu = 1.
    for i in range(max_its):
        num = 2. * num_obs * (np.log(nu) - special.psi(nu)) - dof / nu - dev
        denom = 2. * num_obs * (1. / nu - special.polygamma(1, nu)) + dof / (nu * nu)
        dnu = num / denom
        nu -= dnu * beta
        if abs(dnu) < tol:
            return nu
    else:
        raise ValueError('Could not estimate gamma dispersion.')

class GAM:
    def __init__(self, family=None, link=None, dispersion=None, name=None, load_from_file=None):
        """Generalized Additive Model

        This is the constructor for a Generalized Additive Model.

        References
        ----------
         [glmnet]   glmnet (R package): https://cran.r-project.org/web/packages/glmnet/index.html
                    This is the standard package for GAMs in R and was written by people
                    much smarter than I am!
         [pygam]    pygam (Python package): https://github.com/dswah/pyGAM
                    This is a library in Python that does basically the same thing as this
                    script, but in a different way (not using ADMM).
         [GLM]      Generalized Linear Models by McCullagh and Nelder
                    The standard text on GLMs.
         [GAM]      Generalized Additive Models; by Hastie and Tibshirani
                    The book by the folks who invented GAMs.
         [ESL]      The Elements of Statistical Learning; by Hastie, Tibshirani, and Friedman
                    Covers a lot more than just GAMs.
         [GAMr]     Generalized Additive Models: an Introduction with R; by Wood.
                    Covers more implementation details than [GAM].
         [ADMM]     Distributed Optimization and Statistical Learning via the Alternating
                    Direction Method of Multipliers; by Boyd, Parikh, Chu, Peleato, and Eckstein
                    A mouthful, a work of genius.
         [GAMADMM]  A Distributed Algorithm for Fitting Generalized Additive Models;
                    by Chu, Keshavarz, and Boyd
                    Forms the basis of our approach, the inspiration for this package!

        Parameters
        ----------
         family : str or None (default None)
             Family of the model. Currently supported families include:
                'normal' (for continuous responses),
                'binomial' (for binary responses),
                'poisson' (for counts),
                'gamma' (still in progress),
                'inverse_gaussian' (still in progress).
             Not currently supported families that could be supported
             include Multinomial models (ordinal and nominal) and
             proportional hazards models. Required unless loading an
             existing model from file (see load_from_file).
         link : str or None (optional)
             Link function associated with the model. Supported link
             functions include:
                     Link                Canonical For Family
                'identity'                  'normal'
                'logistic'                  'binomial'
                'log'                       'poisson'
                'reciprocal'                'gamma'
                'reciprocal_squared'        'inverse_gaussian'
             Other links worth supporting include probit, log-log
             and complementary log-log link functions. If not
             specified, the canonical link will be used, but non-
             canonical links are still permitted. Certain link/family
             combinations result in a non-convex problem and
             convergence is not guaranteed.
         dispersion : float or None (optional)
             Dispersion parameter associated with the model. Certain
             families (binomial, poisson) have dispersion independent
             of the data. Specifying the dispersion for these families
             does nothing. In other instances, the dispersion is
             typically unknown and must be estimated from the data.
             If the dispersion is known, it can be specified here which
             will reduce the uncertainty of the model.
         name : str or None (optional)
             Name for model, to be used in plots and in saving files.
         load_from_file : str or None (optional)
             This module uses an iterative approach to fitting models.
             For complicated models with lots of data, each iteration
             can take a long time (though the number of iterations is
             typically less than 100). If the user wishes to pause
             after the end of an iteration, they can pick up where
             the left off by saving results (see the save_flag in .fit)
             and loading them to start the next iterations. Specifying
             this option supercedes all other parameters.

        Returns
        -------
         mdl : Generalized Additive Model object
        """

        if load_from_file is not None:
            self._load(load_from_file)

        if family is None:
            raise ValueError('Family not specified.')
        elif family not in FAMILIES:
            raise ValueError('{} family not supported'.format(family))
        elif family == 'exponential':
            # Exponential is a special case of Gamma with a dispersion of 1.
            self._family = 'gamma'
            dispersion = 1.
        else:
            self._family = family

        if link is None:
            self._link = CANONICAL_LINKS[family]
        elif link in LINKS:
            self._link = link
        else:
            raise ValueError('{} link not supported'.format(link))

        if self._family in FAMILIES_WITH_KNOWN_DISPERSIONS.keys():
            self._known_dispersion = True
            self._dispersion = FAMILIES_WITH_KNOWN_DISPERSIONS[self._family]
        elif dispersion is not None:
            self._known_dispersion = True
            self._dispersion = dispersion
        else:
            self._known_dispersion = False

        if self._link == 'identity':
            self._eval_link = lambda x: x
            self._eval_inv_link = lambda x: x
        elif self._link == 'logistic':
            self._eval_link = lambda x: np.log( x / (1. - x) )
            self._eval_inv_link = lambda x: np.exp(x) / (1 + np.exp(x))
        elif self._link == 'probit':
            self._eval_link = lambda x: stats.norm.ppf(x) # Inverse CDF of the Gaussian distribution
            self._eval_inv_link = lambda x: stats.norm.cdf(x)
        elif self._link == 'complementary_log_log':
            self._eval_link = lambda x: np.log(-np.log(1. - x))
            self._eval_inv_link = lambda x: 1. - np.exp(-np.exp(x))
        elif self._link == 'log':
            self._eval_link = lambda x: np.log(x)
            self._eval_inv_link = lambda x: np.exp(x)
        elif self._link == 'reciprocal':
            self._eval_link = lambda x: 1. / x
            self._eval_inv_link = lambda x: 1. / x
        elif self._link == 'reciprocal_squared':
            self._eval_link = lambda x: 1. / (x * x)
            self._eval_inv_link = lambda x: 1. / np.sqrt(x)

        self._features = {}
        self._offset = 0.0
        self._num_features = 0
        self._fitted = False
        self._name = name

    def _save(self):
        mv = {}
        mv['family'] = self._family
        mv['link'] = self._link
        mv['known_dispersion'] = self._known_dispersion
        if self._known_dispersion:
            mv['dispersion'] = self._dispersion

        mv['offset'] = self._offset
        mv['num_features'] = self._num_features
        mv['fitted'] = self._fitted
        mv['name'] = self._name

        features = {}
        for name, feature in self._features.iteritems():
            if type(feature) == _CategoricalFeature:
                _type = 'categorical'
            elif type(feature) == _LinearFeature:
                _type = 'linear'
            elif type(feature) == _SplineFeature:
                _type = 'spline'
            else:
                raise ValueError('Invalid feature type')

            features[name] = {'type': _type,
                              'filename': feature._filename
                              }

        mv['features'] = features
        mv['f_bar'] = self.f_bar
        mv['z_bar'] = self.z_bar
        mv['u'] = self.u
        mv['prim_res'] = self.prim_res
        mv['dual_res'] = self.dual_res
        mv['prim_tol'] = self.prim_tol
        mv['dual_tol'] = self.dual_tol

        f = open(self._filename, 'w')
        pickle.dump(mv, f)
        f.close()


    def _load(self, filename):
        f = open(filename)
        mv = pickle.load(f)
        f.close()

        self._filename = filename
        self._family = mv['family']
        self._link = mv['link']
        self._known_dispersion = mv['known_dispersion']
        if self._known_dispersion:
            self._dispersion = mv['dispersion']

        self._offset = mv['offset']
        self._num_features = mv['num_features']
        self._fitted = mv['fitted']
        self._name = mv['name']

        self._features = {}
        features = mv['features']
        for (name, feature) in features.iteritems():
            if feature['type'] == 'categorical':
                self._features[name] = _CategoricalFeature(load_from_file=feature['filename'])
            elif feature['type'] == 'linear':
                self._features[name] = _LinearFeature(load_from_file=feature['filename'])
            elif feature['type'] == 'spline':
                self._features[name] = _SplineFeature(load_from_file=feature['filename'])
            else:
                raise ValueError('Invalid feature type')

        self.f_bar = mv['f_bar']
        self.z_bar = mv['z_bar']
        self.u = mv['u']
        self.prim_res = mv['prim_res']
        self.dual_res = mv['dual_res']
        self.prim_tol = mv['prim_tol']
        self.dual_tol = mv['dual_tol']

        if self._link == 'identity':
            self._eval_link = lambda x: x
            self._eval_inv_link = lambda x: x
        elif self._link == 'logistic':
            self._eval_link = lambda x: np.log( x / (1. - x) )
            self._eval_inv_link = lambda x: np.exp(x) / (1 + np.exp(x))
        elif self._link == 'probit':
            self._eval_link = lambda x: stats.norm.ppf(x) # Inverse CDF of the Gaussian distribution
            self._eval_inv_link = lambda x: stats.norm.cdf(x)
        elif self._link == 'complementary_log_log':
            self._eval_link = lambda x: np.log(-np.log(1. - x))
            self._eval_inv_link = lambda x: 1. - np.exp(-np.exp(x))
        elif self._link == 'log':
            self._eval_link = lambda x: np.log(x)
            self._eval_inv_link = lambda x: np.exp(x)
        elif self._link == 'reciprocal':
            self._eval_link = lambda x: 1. / x
            self._eval_inv_link = lambda x: 1. / x
        elif self._link == 'reciprocal_squared':
            self._eval_link = lambda x: 1. / (x * x)
            self._eval_inv_link = lambda x: 1. / np.sqrt(x)


    def add_feature(self, name, type, transform=None, rel_dof=None, regularization=None):
        """Add a feature

        Add a feature to a Generalized Additive Model. (An implicit
        constant feature is always included, representing the overall
        average response.)

        Parameters
        ----------
         name : str
             Name for feature. Used internally to keep track of
             features and is also used when saving files and in
             plots.
         type : str
             Type of feature. Currently supported options include:
               'categorical' (for categorical variables)
               'linear' (for variables with a linear contribution
                         to the response)
               'spline' (for variables with a potentially nonlinear
                         contribution to the response).
             Other types of features worth supporting include
             piecewise constant functions and monotonic functions.
             Those might end up being regularization terms.
         transform : function or None
             Optional transform applied to feature data, saving
             the user from repetitive boilerplate code. Any function
             may be used; it is applied to data provided during fitting
             and prediction. Common options might include np.log, np.log1p,
             or np.sqrt. The user may wish to start with a base feature
             like 'age' and use derived features 'age_linear', 'age_quadratic'
             to permit quadratic models for that feature, with potentially
             different regularization applied to each.
         rel_dof : float or None
             Relative degrees of freedom. Applicable only to spline features.
             The degrees of freedom associated with a spline represent how
             "wiggly" it is allowed to be. A spline with two degrees of freedom
             is just a line. (Actually, since these features are constrained
             to have zero mean response over the data, linear features
             only have one degree of freedom.) The relative degrees of freedom
             are used to specify the baseline smoothing parameter (lambda)
             associated with a feature. When the model is fit to data, the user
             can specify an overall smoothing parameter applied to all features
             to alter the amount of regularization in the entire model. Thus
             the actual degrees of freedom will vary based on the amount of
             smoothing. The idea is that the analyst may wish to permit some
             features to be more wiggly than others. By default, all
             splines have 4 relative degrees of freedom.

             Regularization of any feature effectively reduces the degrees of
             freedom, and so this term is potentially applicable, but that is
             not yet supported.
        regularization : dictionary or None
             Dictionary specifying the regularization applied to this feature.
             Different types of features support different types of regularization.
             Splines implicitly only support regularization of the wiggliness
             via a C2 smoothness penalty. That is controlled via the rel_dof.
             Other features have more diverse options described in their own
             documentation.

        Returns
        -------
         (nothing)

        """
        if type == 'categorical':
            f = _CategoricalFeature(name, regularization=regularization)
        elif type == 'linear':
            f = _LinearFeature(name, transform, regularization=regularization)
        elif type == 'spline':
            f = _SplineFeature(name, transform, rel_dof)
        else:
            raise ValueError('Features of type {} not supported.'.format(type))

        self._features[name] = f
        self._num_features += 1

    def fit(self, X, y, weights=None, optimizer='admm', smoothing=1., save_flag=False, verbose=False, plot_convergence=False, max_its=100):
        """Fit a Generalized Additive Model to data.

        Note regarding binomial families: many data sets include
        multiple observations having identical features. For example,
        imagine a data set with features 'gender', and 'country' and
        binary response indicating whether the person died (morbid
        but common in biostatistics). The data might look like this:

           gender   country   patients   survivors
             M        USA       50           48
             F        USA       70           65
             M        CAN       40           38
             F        CAN       45           43

        This still describes a binomial family, but in a more compact
        format than specifying each individual user. We eventually
        want to support this more compact format, but we do not
        currently! In this context, it is important to check for
        over-dispersion (see [GLM]), and I need to learn more first.
        In the current implementation, we assume that there is no
        over-dispersion, and that the number of users having the
        same set of features is small.

        Parameters
        ----------
         X : pandas dataframe
             Dataframe of features. The column names must correspond
             to the names of features added to the model. X may have
             extra columns corresponding to features not included in
             the model; these are simply ignored. Where applicable,
             the data should be "pre-transformation", since this code
             will apply any transformations specified in .add_feature.
         y : array
             Response. Depending on the model family, the response
             may need to be in a particular form (for example, for
             a binomial family, the y's should be either 0 or 1),
             but this is not checked anywhere!
         w : array
             Weights applied to each observation. This is effectively
             specifying the dispersion of each observation.
         optimizer : string
             We use the Alternating Direction Method of Multipliers
             ('admm') to fit the model. We may eventually support more
             methods, but right now this option does nothing.
         smoothing : float
             Smoothing to apply to entire model, used in conjunction
             with other regularization parameters. That is, whatever
             regularization is used for the various features, is
             scaled by this term, allowing the user to set the overall
             smoothing by Cross Validation or whatever they like. This
             allows the user to specify different regularization for
             each feature, while still permitting a one-dimensional
             family of models corresponding to different amounts of
             regularization. Defaults to 1., leaving the regularization
             as specified in .add_feature().
         save_flag : boolean
             Specifies whether to save intermediate results after each
             iteration. Useful for complicated models with massive
             data sets that take a while to fit. If the system crashes
             during the fit, the analyst can pick up where they left
             off instead of starting from scratch. Defaults to False.
         verbose : boolean
             Specifies whether to print mildly useful information to
             the screen during the fit. Defaults to False.
         plot_convergence : boolean
             Specifies whether to plot the convergence graph at the
             end. (I suspect only Convex Optimization nerds like me
             want to see this.) Defaults to False.
         max_its : integer
             Maximum number of iterations. Defaults to 100.

        Returns
        -------
         (nothing)

        """
        if save_flag and self._name is None:
            raise ValueError('Cannot save a GAM with no name. Specify name when instantiating model.')

        if len(X) != len(y):
            raise ValueError('Inconsistent number of observations in X and y.')

        num_threads = 1
        self._rho = 0.1
        eps_abs = 1e-3
        eps_rel = 1e-3
        # Note that X may include columns that do not correspond to features in our model
        # (for example, if the user is experimenting with leaving out features to assess
        # importance). Thus, the real number of features is self._num_features, not
        # num_features as in the next line.
        self._num_obs, num_features = X.shape

        self._y = y.flatten()
        self._weights = weights
        self._offset = self._eval_link(np.mean(self._y))
        self.fj = {}

        for name, feature in self._features.iteritems():
            feature.initialize(X[name].values, smoothing=smoothing, save_flag=save_flag, save_prefix=self._name)
            self.fj[name] = np.zeros(self._num_obs)

        self.f_bar = np.full((self._num_obs,), self._offset / self._num_features)
        self.z_bar = np.zeros(self._num_obs)
        self.u = np.zeros(self._num_obs)
        self.prim_res = []
        self.dual_res = []
        self.prim_tol = []
        self.dual_tol = []
        self.dev = []

        z_new = np.zeros(self._num_obs)

        if num_threads > 1:
            p = mp.Pool(num_threads)
        else:
            p = None

        for i in range(max_its):
            if verbose:
                print 'Iteration {0:d}'.format(i)
                print 'Optimizing primal variables'

            fpumz = self._num_features * (self.f_bar + self.u - self.z_bar)
            self.fj_new = {}
            f_new = np.full((self._num_obs,), self._offset)
            if False: #num_threads > 1:
                # Getting python to run a for loop in parallel
                # might as well be impossible :-(
                args = [(i, self._features[i], fpumz, self._rho) for i in self._features.keys()]
                results = p.map(_feature_wrapper, args)
                for i in results:
                    self.fj_new[i[0]] = i[1]
                    f_new += i[1]

            else:
                for name, feature in self._features.iteritems():
                    if verbose:
                          print 'Optimizing {0:s}'.format(name)
                    self.fj_new[name] = feature.optimize(fpumz, self._rho)
                    f_new += self.fj_new[name]

            f_new /= self._num_features

            if verbose:
                print 'Optimizing dual variables'

            z_new = self._optimize(self.u + f_new, self._num_features, p)

            self.u += f_new - z_new

            prim_res = np.sqrt(self._num_features) * linalg.norm(f_new - z_new)
            dual_res = 0.0
            norm_ax = 0.0
            norm_bz = 0.0
            norm_aty = 0.0
            num_params = 0
            for name, feature in self._features.iteritems():
                dr = (self.fj_new[name] - self.fj[name]) + (z_new - self.z_bar) - (f_new - self.f_bar)
                dual_res += dr.dot(dr)
                norm_ax += self.fj_new[name].dot(self.fj_new[name])
                zik = self.fj_new[name] + z_new - f_new
                norm_bz += zik.dot(zik)
                norm_aty += feature.compute_dual_tol(self.u)
                num_params += feature.num_params()

            dual_res = self._rho * np.sqrt(dual_res)
            norm_ax = np.sqrt(norm_ax)
            norm_bz = np.sqrt(norm_bz)
            norm_aty = np.sqrt(norm_aty)

            self.f_bar = f_new
            self.fj = self.fj_new
            self.z_bar = z_new
            prim_tol = np.sqrt(self._num_obs * self._num_features) * eps_abs + eps_rel * np.max([norm_ax, norm_bz])
            dual_tol = np.sqrt(num_params) * eps_abs + eps_rel * norm_aty

            self.prim_res.append(prim_res)
            self.dual_res.append(dual_res)
            self.prim_tol.append(prim_tol)
            self.dual_tol.append(dual_tol)
            self.dev.append(self.deviance())

            if prim_res < prim_tol and dual_res < dual_tol:
                if verbose:
                    print 'Fit converged'
                break
        else:
            if verbose:
                print 'Fit did not converge'

        if num_threads > 1:
            p.close()
            p.join()

        self._fitted = True

        if plot_convergence:
            _plot_convergence(self.prim_res, self.prim_tol, self.dual_res, self.dual_tol, self.dev)

    def _optimize(self, upf, N, p=None):
        """Optimize \bar{z}.

        Solves the optimization problem:
           minimize L(N*z) + \rho/2 * \| N*z - N*u - N*\bar{f} \|_2^2
        where z is the variable, N is the number of features, u is the scaled
        dual variable, \bar{f} is the average feature response, and L is
        the likelihood function which is different depending on the
        family and link function. This is accomplished via a proximal
        operator, as discussed in [GAMADMM]:
          prox_\mu(v) := argmin_x L(x) + \mu/2 * \| x - v \|_2^2
        I strongly believe that paper contains a typo in this equation, so we
        return (1. / N) * prox_\mu (N * (u + \bar{f}) with \mu = \rho instead
        of \mu = \rho / N as in [GAMADMM]. When implemented as in the paper,
        convergence was much slower, but it did still converge.

        Certain combinations of family and link function result in proximal
        operators with closed form solutions, making this step *very* fast
        (e.g. 3 flops per observation).

        Parameters
        ----------
         upf : array
             Vector representing u + \bar{f}
         N : integer
             Number of features.
         p : Multiprocessing Pool (optional)
             If multiple threads are available, massive data sets may
             benefit from solving this optimization problem in parallel.
             It is up to the individual functions to decide whether to
             actually do this.

        Returns
        -------
         z : array
             Result of the above optimization problem.
        """

        prox = None
        if self._family == 'normal':
            if self._link == 'identity':
                prox = po._prox_normal_identity
            else:
                prox = po._prox_normal
        elif self._family == 'binomial':
            if self._link == 'logistic':
                prox = po._prox_binomial_logit
            else:
                prox = po._prox_binomial
        elif self._family == 'poisson':
            if self._link == 'log':
                prox = po._prox_poisson_log
            else:
                prox = po._prox_poisson
        elif self._family == 'gamma':
            if self._link == 'reciprocal':
                prox = po._prox_gamma_reciprocal
            else:
                prox = po._prox_gamma
        elif self._family == 'inverse_gaussian':
            if self._link == 'reciprocal_squared':
                prox = po._prox_inv_gaussian_reciprocal_squared
            else:
                prox = po._prox_inv_gaussian
        else:
            raise ValueError('Family {0:s} and Link Function {1:s} not (yet) supported.'.format(self._family, self._link))

        return (1. / N) * prox(N*upf, self._rho, self._y, self._weights, self._eval_inv_link, p)

    def predict(self, X):
        """Apply fitted model to features.

        Parameters
        ----------
         X : pandas dataframe
             Data for which we wish to predict the response. The
             column names must correspond to the names of the
             features used to fit the model. X may have extra
             columns corresponding to features not in the model;
             these are simply ignored. Where applicable, the data
             should be "pre-transformation", since this code will
             apply any transformations specified while defining
             the model.

        Returns
        -------
         mu : array
             Predicted mean response for each data point.

        """
        if not self._fitted:
            raise AttributeError('Model not yet fit.')

        num_points, m = X.shape
        eta = np.full((num_points,), self._offset)
        for name, feature in self._features.iteritems():
            eta += feature.predict(X[name].values)

        return self._eval_inv_link(eta)

    def confidence_intervals(self, X, prediction=False, width=0.95):
        """Confidence intervals on predictions.

        NOT YET IMPLEMENTED

        There are two notions of confidence intervals that are
        appropriate. The first is a confidence interval on mu,
        the mean response. This follows from the uncertainty
        associated with the fit model. The second is a confidence
        interval on observations of this model. The distinction
        is best understood by example. For a Gaussian family,
        the model might be a perfect fit to the data, and we
        may have billions of observations, so we know mu perfectly.
        Confidence intervals on the mean response would be very
        small. But the response is Gaussian with a non-zero
        variance, so observations will in general still be spread
        around the mean response. A confidence interval on the
        prediction would be larger.

        Now consider a binomial family. The estimated mean response
        will be some number between 0 and 1, and we can estimate
        a confidence interval for that mean. But the observed
        response is always either 0 or 1, so it doesn't make sense
        to talk about a confidence interval on the prediction
        (except in some pedantic sense perhaps).

        Note that if we are making multiple predictions, it makes
        sense to talk about a "global" set of confidence intervals.
        Such a set has the property that *all* predictions fall
        within their intervals with specified probability. This
        function does not compute global confidence intervals!
        Instead each confidence interval is computed "in vacuo".

        Parameters
        ----------
         X : pandas dataframe
             Data for which we wish to predict the response. The
             column names must correspond to the names of the
             features used to fit the model. X may have extra
             columns corresponding to features not in the model;
             these are simply ignored. Where applicable, the data
             should be "pre-transformation", since this code will
             apply any transformations specified while defining
             the model.
         prediction : boolean
             Specifies whether to return a confidence interval
             on the mean response or on the predicted response.
             (See above.) Defaults to False, leading to a
             confidence interval on the mean response.
         width : float between 0 and 1
             Desired confidence width. Defaults to 0.95.

        Returns
        -------
         mu : (n x 2) array
             Lower and upper bounds on the confidence interval
             associated with each prediction.
        """
        pass

    def plot(self, name, true_fn=None):
        """Plot the component of the modelf for a particular feature.

        Parameters
        ----------
         name : str
             Name of feature (must be a feature in the model).
         true_fn : function or None (optional)
             Function representing the "true" relationship
             between the feature and the response.

        Returns
        -------
         (nothing)

        """
        self._features[name]._plot(true_fn=true_fn)

    def deviance(self, X=None, y=None, w=None):
        """Deviance

        This function works in one of two ways:

        Firstly, it computes the deviance of the model, defined as
           2 * \phi * (\ell(y; y) - \ell(\mu; y))
        where \phi is the dispersion (which is only in this equation
        to cancel out the denominator of the log-likelihood),
        \ell(y; y) is the log-likelihood of the model that fits the
        data perfectly, and \ell(\mu; y) is the log-likelihood of the
        fitted model on the data used to fit the model. This is
        the quantity we minimize when fitting the model.

        Secondly, it computes the deviance of the model on arbitrary
        data sets. This can be used in conjunction with Cross Validation
        to choose the smoothing parameter by minimizing the deviance
        on the hold-out set.

        Parameters
        ----------
         X : pandas dataframe (optional)
             Dataframe of features. The column names must correspond
             to the names of features added to the model. (See .predict()).
             Only applicable for the second use case described above.
         y : array (optional)
             Response. Only applicable for the second use case.
         w : array (optional)
             Weights for observations. Only applicable for the second
             use case, but optional even then.

        Returns
        -------
         D : float
             The deviance of the model.
        """
        if X is None or y is None:
            y = self._y
            mu = self._eval_inv_link(self._num_features * self.f_bar)
            w = self._weights
        else:
            mu = self.predict(X)

        if self._family == 'normal':
            y_minus_mu = y - mu
            if w is None:
                return y_minus_mu.dot(y_minus_mu)
            else:
                return w.dot(y_minus_mu * y_minus_mu)
        elif self._family == 'binomial':
            if w is None:
                return -2. * np.sum( y * np.log(mu) + (1. - y) * np.log1p(-mu) )
            else:
                return -2. *  w.dot( y * np.log(mu) + (1. - y) * np.log1p(-mu) )
        elif self._family == 'poisson':
            if w is None:
                return 2. * np.sum(y * np.log(y / mu) - (y - mu))
            else:
                return 2. * w.dot(y * np.log(y / mu) - (y - mu))
        elif self._family == 'gamma':
            if w is None:
                return 2. * np.sum(-1. * np.log(y / mu) + (y - mu) / mu)
            else:
                return 2. * w.dot(-1. * np.log(y / mu) + (y - mu) / mu)
        elif self._family == 'inverse_gaussian':
            if w is None:
                return np.sum( (y - mu) * (y - mu) / (mu * mu * y) )
            else:
                return w.dot( (y - mu) * (y - mu) / (mu * mu * y) )

    def dispersion(self, formula='deviance'):
        """Dispersion

        Returns the dispersion associated with the model. Depending on
        the model family and whether the dispersion was specified by the
        user, the dispersion may or may not be known a priori. This
        function will estimate this parameter when appropriate.

        There are different ways of estimating this parameter that may
        be appropriate for different kinds of families. The current
        implementation is based on the deviance, as in Eqn 3.10 on
        p. 110 of GAMr. As discussed in that section, this tends not to
        work well for Poisson data (with overdispersion) when the mean
        response is small. Alternatives are offered in that section, but
        I have not yet implemented them. This is not terribly relevant
        for the current implementation since overdispersion is not
        supported! (When overdispersion is not present, the dispersion
        of the Poisson is exactly 1.)

        My eventual hope is to understand the appropriate methods for
        all the different circumstances and have intelligent defaults
        that can be overridden by opinionated users.

        Parameters
        ----------
         formula : str
             Formula for the dispersion. Options include:
                'deviance' (default)
                'pearson'
                'fletcher'
        """
        if self._family == 'normal':
            if self._known_dispersion:
                return self._dispersion
            else:
                sigma2 = self.deviance() / (self._num_obs - self.dof())
                return sigma2
        elif self._family == 'binomial':
            return 1.
        elif self._family == 'poisson':
            return 1.
        elif self._family == 'gamma':
            if self._known_dispersion:
                return self._dispersion
            else:
                return _gamma_dispersion(self.dof(), self.deviance(), self._num_obs)
                # This equation is a first-order approximation valid when nu is
                # large (see Section 8.3.6 of [GLM])
                #Dbar = self.deviance() / self._num_obs
                #return Dbar * (6. + Dbar) / (6. + 2. * Dbar)
        elif self._family == 'inverse_gaussian':
            if self._known_dispersion:
                return self._dispersion
            else:
                sigma2 = self.deviance() / (self._num_obs - self.dof())
                return sigma2

    def dof(self):
        """Degrees of Freedom

        Returns the degrees of freedom associated with this model.
        Simply adds up the degrees of freedom associated with each
        feature.
        """
        dof = 1. # Affine factor
        for name, feature in self._features.iteritems():
            dof += feature.dof()
        return dof

    def aic(self):
        """Akaike Information Criterion

        Returns the AIC for the fitted model, useful for choosing
        smoothing parameters. The AIC we compute is actually off
        by a constant factor, making it easier to compute without
        detracting from its role in model selection.

        Different authors seem to throw in multiplicative or additive
        factors willy-nilly since it doesn't affect model selection.
        """
        p = self.dof()
        if not self._known_dispersion:
            # If we are estimating the dispersion, we need to
            # add one to the DOF.
            p += 1

        # Note that the deviance is twice the dispersion times the log-likelihood, so no factor
        # of two required there.
        return self.deviance() / self.dispersion() + 2. * p
        #return self.deviance() / self._num_obs + 2. * p * self.dispersion() / self._num_obs

    def aicc(self):
        # Eqn 6.32 on p. 304 of [GAMr]
        pass

    def ubre(self, gamma=1.0):
        """Un-Biased Risk Estimator

        Returns the Un-Biased Risk Estimator as discussed in Sections
        6.2.1 and 6.2.5 of [GAMr]. This can be used for choosing the
        smoothing parameter when the dispersion is known.

        As discussed in Section 6.2.5 of [GAMr], sometimes it is helpful
        to force smoother fits by exaggerating the effective degrees of
        freedom. In that case, a value of gamma > 1. may be desirable.
        """
        return self.deviance() + 2. * gamma * self.dispersion() * self.dof()

    def gcv(self, gamma=1.0):
        """Generalized Cross Validation

        This function returns the Generalized Cross Validation (GCV)
        score, which can be used for choosing the smoothing parameter
        when the dispersion is unknown.

        As discussed in Section 6.2.5 of [GAMr], sometimes it is helpful
        to force smoother fits by exaggerating the effective degrees of
        freedom. In that case, a value of gamma > 1. may be desirable.
        """
        return self._num_obs * self.deviance() / ((self._num_obs - gamma * self.dof()) ** 2.)

    def summary(self):
        """Print summary statistics associated with fitted model.

        Prints statistics for the overall model, as well as for
        each individual feature (see the __str__() function in
        each feature type for details about what is printed
        there).

        For the overall model, the following are printed:
           phi:      Estimated dispersion parameter. Omitted
                     if specified or if it is known for the
                     Family (e.g. Poisson).
           edof:     Estimated degrees of freedom.
           Deviance: The difference between the log-likelihood of
                     the model that fits the data perfectly and
                     that of the fitted model, times twice the
                     dispersion.
           AIC:      Akaike Information Criterion.
           AICc:     AIC with correction for finite data sets.
           UBRE:     Unbiased Risk Estimator (if dispersion is known).
           GCV:      Generalized Cross Validation (if dispersion is estimated).

        For more details on these parameters, see the documentation
        in the corresponding functions. It may also be helpful to
        include an R^2 value where appropriate, and perhaps a p-value
        for the model against the null model having just the affine
        term. It would also be nice to have confidence intervals
        at least on the estimated dispersion parameter.
        """

        print 'Model Statistics'
        print '----------------'
        if not self._known_dispersion:
            print 'phi: {0:0.06g}'.format(self.dispersion())
        print 'edof: {0:0.0f}'.format(self.dof())
        print 'Deviance: {0:0.06g}'.format(self.deviance())
        print 'AIC: {0:0.06g}'.format(self.aic())
        #print 'AICc: {0:0.06g}'.format(aicc)

        if self._known_dispersion:
            print 'UBRE: {0:0.06g}'.format(self.ubre())
        else:
            print 'GCV: {0:0.06g}'.format(self.gcv())

        print ''
        print 'Features'
        print '--------'

        for name, feature in self._features.iteritems():
            print feature.__str__()


    def _save(self):
        '''Save state.'''
        mv = {}
        mv['f'] = self.f_bar
        mv['z'] = self.z_bar
        mv['u'] = self.u
        mv['prim_res'] = self.prim_res
        mv['dual_res'] = self.dual_res
        mv['prim_tol'] = self.prim_tol
        mv['dual_tol'] = self.dual_tol
        mv['name'] = self._name
        mv['save_self'] = self._save_self

        f = open('{0:s}.pckl'.format(self._name), 'w')
        pickle.dump(mv, f)
        f.close()

    def _load(self, filename):
        '''Load parameters from a previous model fitting session.'''
        f = open(filename)
        mv = pickle.load(f)
        f.close()

        self.f_bar = mv['f']
        self.z_bar = mv['z']
        self.u = mv['u']
        self.prim_res = mv['prim_res']
        self.dual_res = mv['dual_res']
        self.prim_tol = mv['prim_tol']
        self.dual_tol = mv['dual_tol']
        self._name = mv['name']
        self._save_self = mv['save_self']
