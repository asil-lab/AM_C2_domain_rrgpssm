import numpy as np
import utils
from numpy.random import normal, uniform
from numpy.linalg import cholesky, pinv, norm, det
import scipy.stats as sps
from scipy.stats import multivariate_normal as mvn
from numpy.matlib import repmat
import pathlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl
font = {'size': 14}
mpl.rc('font', **font)
mpl.use('Qt5Agg')
cwd = os.getcwd()
plt.ion()
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

'''
    Python code for paper 'Mishra et al, Domain-aware Gaussian process state-space models, Signal Processing 238, pp. 110003, 2026.
    DOI: https://doi.org/10.1016/j.sigpro.2025.110003
    
    NOTE: Variable names used here closely match the those used in the paper.
    
    Domain-Aware Gaussian Process State-Space Models (DA-GPSSMs), based on Gaussian Processe State-Space Models,
    use domain information, in the form of interatcions between state dimensions, to reduce computational load
    while learning in highly nonlinear state-space models. We use Lotka-Volterra model, used to model populations 
    dynamics of multi-species predator-prey ecosystem, to showcase the computational performance of DA-GPSSMs.
    
    Essential parameters determining the computational load:
    T: Length of the timeseries
    D: Dimensions of the state-space
    M: Truncation order for spectral approximation of the GP
    N = Number of particles used in the filter
    K: Number of iterations for the particle filter
    k_mh: Number of iteration for Metropolis_Hastings algorithm
     
    Author:
    Anurodh Mishra
    Delft University of Technology
    Delft, The Netherlands
'''

'''
    Functions for use within the code
'''
# Simulating system dynamics
def lv_dynamics(x, alpha=np.array([3, 4, 7.2]), beta=np.array([[-0.5, -1, 0], [0, -1, -2], [-2.6, -1.6, -3]])):
    Dx = len(x)
    Delx = np.zeros(Dx)
    for ii in range(Dx):
        Delx[ii] = alpha[ii] * x[ii]
        for jj in range(Dx):
            Delx[ii] += beta[ii, jj] * x[ii] * x[jj]

    return Delx

# Eigenvalues of the SE kernel
def se_kernel_eigvals(m, l):
    d = l.shape[2]
    jv = np.zeros((m ** d, 1, d))
    eigvals = np.zeros((m ** d, d))
    if d == 3:
        for ii in range(1, m + 1):
            for jj in range(1, m + 1):
                for kk in range(1, m + 1):
                    ind = m ** 2 * (ii - 1) + m * (jj - 1) + kk
                    jv[ind - 1, 0, :] = np.array([ii, jj, kk])
                    eigvals[ind - 1, :] = np.power(np.pi * jv[ind - 1, 0, :] / (2. * np.squeeze(l)), 2)
    elif d == 2:
        for jj in range(1, m + 1):
            for kk in range(1, m + 1):
                ind = m * (jj - 1) + kk
                jv[ind - 1, 0, :] = np.array([jj, kk])
                eigvals[ind - 1, :] = np.power(np.pi * jv[ind - 1, 0, :] / (2. * np.squeeze(l)), 2)
    else:
        raise Exception("This dimensionality has not been implemented!")

    return jv, eigvals

# Eigenfunction of the SE kernel
phi_x = lambda x, idx: np.prod(np.multiply(np.power(L[idx], -1 / 2), np.sin(np.pi * np.multiply(np.multiply(jv[idx], (np.reshape(x, (1, 1, L[idx].shape[2])) + L[idx])), 1. / (2 * L[idx])))), axis=2)

# Prior for column covariance of A using SE kernel
S_SE = lambda w, l: np.sqrt(2 * np.pi * l ** 2) * np.exp(-(w ** 2 / 2.) * l ** 2)
V_prior = S_f = lambda lsx, varx, idx: 1 / varx * np.diag(np.prod(1. / S_SE(np.sqrt(lmd[idx]), np.tile(lsx, (M_bar[idx], 1))), 1))

# Priors for kernel hyperparaemters theta = {ls, var} i.e. (lengthscale, variance)
ls_prior = lambda lsx, idx: utils.mvnpdf_log(np.log(lsx), np.squeeze(repmat(np.log(10), len(dims[idx]), 1)), np.identity(len(dims[idx])))
var_prior = lambda varx: utils.mvnpdf_log(varx, 100, 100)

'''
    General simulation parameters
'''
# Set random number generator(for reproducible results)
SEED = 1
np.random.seed(SEED)

# Number of iterations to be performed
K = 30

# Flags for noise, plots and save
NOISE = True
SAVE = False

'''
    System dynamics simulation
'''
# Time parameters
tend = 40
dt = 0.1
t = np.arange(0, tend + dt, dt)
T = len(t)

# Number of state dimensions
D = 3

# Variable allocation
Xt = np.zeros((T, D)) # noiseless trajectory
X = np.zeros((T, D)) # noisy trajectory
Y = np.zeros((T, D)) # noisy measurement

# Initializations for path
X[0, :] = np.array([0.1, 0.8, 0.3])
Xt[0, :] = np.array([0.1, 0.8, 0.3])

# Measurement model g
g = lambda x: x

# noise parameters
Q = 0.01 * np.eye(D) # process covariance
R = 0.01 * np.eye(D) # measurement covariance

# Simulate trajectory
for tt in range(1, len(t)):
    Xt[tt, :] = Xt[tt - 1, :] + dt * lv_dynamics(Xt[tt - 1, :])
    X[tt, :] = Xt[tt - 1, :] + dt * lv_dynamics(X[tt - 1, :]) + NOISE * mvn.rvs(mean=np.zeros(D), cov=Q)
    Y[tt, :] = g(X[tt, :]) + NOISE * mvn.rvs(mean=np.zeros(D), cov=R)
Y[0, :] = g(X[0, :]) + NOISE * mvn.rvs(mean=np.zeros(D), cov=R)

'''
    Simulation parameters for DA-GPSSMs
'''
# Number of particles
N = 5

# Number of iterations for the Metropolis-Hastings sampler
K_mh = 5

# Number of MCMC experiments
NEXP = 100

# Priors for process covariance Q
lQ = 1 # defree of freedom
LambdaQ = 1 * np.identity(D) # scale matrix

# Truncation order, i.e. number of terms in spectral approximation
M = 6

# Dimensional dependency and boundary conditions
dims = [[0, 1], [1, 2], [0, 1, 2]]
L = [np.array([[[4., 5.]]]), np.array([[[5., 2.5]]]), np.array([[[4., 5., 2.5]]])]

# Number of coefficients to learn for each dimension
M_bar = np.zeros(D, dtype=int)
for dd in range(D):
    M_bar[dd] = M ** len(dims[dd])

# Eigenvalues of the SE kernel
jv = [None] * D
lmd = [None] * D
for dd in range(D):
    indices, eigvals = se_kernel_eigvals(M, L[dd])
    jv[dd] = indices
    lmd[dd] = eigvals

# Pre-allocate and initialize variables of interest for all experiments
Q = np.zeros((D, D, K+1, NEXP))
A0 = np.zeros((1, M_bar[0], K + 1, NEXP))
A1 = np.zeros((1, M_bar[1], K + 1, NEXP))
A2 = np.zeros((1, M_bar[2], K + 1, NEXP))
ls0 = np.zeros((len(dims[0]), K + 1, NEXP))
ls1 = np.zeros((len(dims[1]), K + 1, NEXP))
ls2 = np.zeros((len(dims[2]), K + 1, NEXP))
var = np.zeros((D, K + 1, NEXP))
x_prim = np.zeros((D, T, K + 1, NEXP))
x_pf = np.zeros((D, N, T, K, NEXP))
err_norm = np.zeros((K + 1, D, NEXP))

'''
    MCMC learning algorithm for DA-GPSSMs
'''
# trajectory variable for storing paths of each particle
for iter in range(NEXP):
    SEED = 1 + iter
    np.random.seed(SEED)

    # Creating the local copies of variables per experiment and initializing
    A = [None] * D
    ls = [None] * D
    for dd in range(D):
        A[dd] = np.zeros((1, M_bar[dd], K + 1))
        A[dd][0, :, 0] = 10 * normal(size=M_bar[dd])
        ls[dd] = np.zeros((len(dims[dd]), K + 1))
        ls[dd][:, 0] = uniform(size=len(dims[dd]))
        err_norm[0, dd, iter] = norm(Xt[:, dd] - x_prim[dd, :, 0, iter])
    Q[:, :, 0, iter] = 1 * np.identity(D)
    var[:, 0, iter] = 1 * np.ones(D)

    for kk in range(K):
        # Evaluating the GP function value for each iteration
        f_i = lambda x, idx: A[idx][:, :, kk] @ phi_x(x, idx).reshape(M_bar[idx], 1)
        Q_chol = cholesky(Q[:, :, kk, iter])

        '''
            Conditional particle filter with ancestral sampling (CPF-AS)
        '''
        # Pre - allocate for weight and ancestors to be used in CPF-AS
        weights = np.zeros((T, N))
        ancestors = np.zeros((T, N))

        # Initialize
        if kk > 0:
            x_pf[:, -1, :, kk] = x_prim[:, :, kk]
        weights[0, :] = 1
        weights[0, :] = weights[0, :] / np.sum(weights[0, :])

        # Using the output measurement at t=0 to initialize
        x_pf[:, 0: -1, 0, kk, iter] = np.tile(Y[0, :].reshape(D, 1), (1, N - 1))
        x_pf[:, -1, 0, kk, iter] = x_prim[:, 0, kk, iter]

        # for each time step in the timeseries
        for tt in range(T):
            if tt >= 1:
                # sampling ancestor for each particle
                ancestors[tt, :-1] = utils.systematic_resampling(weights[tt - 1, :], N - 1)

                # updating time step
                for nn in range(N - 1):
                    for dd in range(D):
                        x_bar = x_pf[:, int(ancestors[tt, nn]), tt - 1, kk, iter]
                        x_pf[dd, nn, tt, kk, iter] = np.squeeze(f_i(x_bar[dims[dd]], dd) + Q_chol[dd, dd] * normal(size=1))

                # Initializing the weights and the last particle location
                weight_N = np.zeros(N)
                xN = np.zeros(D)
                for nn in range(N):
                    for dd in range(D):
                        x_bar = x_pf[:, nn, tt - 1, kk, iter]
                        xN[dd] = np.squeeze(f_i(x_bar[dims[dd]], dd))
                    weight_N[nn] = weights[tt - 1, nn] * mvn.pdf(x_pf[:, -1, tt, kk, iter], mean=xN, cov=Q[:, :, kk, iter])
                weight_N = weight_N / np.sum(weight_N)
                ancestors[tt, -1] = utils.systematic_resampling(weight_N, 1)

            # PF weight update
            for nn in range(N):
                weights[tt, nn] = mvn.pdf(x_pf[:, nn, tt, kk, iter], mean=Y[tt, :], cov=R)
            weights[tt, :] = weights[tt, :] / np.sum(weights[tt, :])

        # Sample trajectory to condition on
        star = utils.systematic_resampling(weights[-1, :], 1)
        x_prim[:, -1, kk + 1, iter] = x_pf[:, star, -1, kk, iter]
        for tt in np.arange(T-2, -1,  -1):
            star = ancestors[tt + 1, int(star)]
            x_prim[:, tt, kk + 1, iter] = x_pf[:, int(star), tt, kk, iter]

        print('Sampling. k = ' + str(kk + 1) + '/' + str(K) + ' of Exp: ' + str(iter + 1) + '/' + str(NEXP))

        '''
            Posterior distributions of coefficient matrix A, process covariance Q and kernel hyperparameters theta
            per dimension
        '''
        for dd in range(D):
            Z = np.expand_dims(x_prim[dd, 1: T, kk + 1, iter].reshape(1, T - 1), axis=1)
            Phi = np.zeros((M_bar[dd], 1, T - 1))
            for tt in range(T - 1):
                x_bar = x_prim[:, tt, kk + 1, iter]
                Phi[:, 0, tt] = np.squeeze(phi_x(x_bar[dims[dd]], dd))

            ZZT = np.sum(np.asarray(np.tile(Z, (1, 1, 1))) * Z.swapaxes(0, 1), axis=2)
            ZPhiT = np.sum(np.asarray(np.tile(Z, (1, M_bar[dd], 1))) * Phi.swapaxes(0, 1), axis=2)
            PhiPhiT = np.sum(np.asarray(np.tile(Phi, (1, M_bar[dd], 1))) * Phi.swapaxes(0, 1), axis=2)

            # Sampling prior column covariance of A
            V = V_prior(ls[dd][:, kk], var[0, kk, iter], dd)

            # NOTE: Update the value if GP is not zero-mean (see Wills et al., Estimation of Linear Systems using a Gibbs Sampler, 2012)
            PhiPhiT_bar = PhiPhiT + V
            PhiPhiT_bar = 0.5 * (PhiPhiT_bar + PhiPhiT_bar.T)
            ZPhiT_bar = ZPhiT
            ZZT_bar = ZZT

            # Computing the posterior mean of matrix A
            PhiPhiT_bar_inv = pinv(PhiPhiT_bar)
            A_mean = ZPhiT_bar @ PhiPhiT_bar_inv

            # Computing the posterior scale of matrix Q
            Q_scale = LambdaQ[dd, dd] + ZZT_bar - A_mean @ ZPhiT_bar.T
            Q_scale = 0.5 * (Q_scale + Q_scale.T)

            # Sampling from posterior p(Q | x_{0:T})
            Q[dd, dd, kk + 1] = sps.invwishart.rvs(df=T - 1 + lQ, scale=Q_scale)

            # Sampling from posterior p(A | Q, x_{0:T})
            x = normal(size=(1, M_bar[dd]))
            A[dd][:, :, kk + 1] = A_mean + cholesky(Q[dd, dd, kk + 1, iter].reshape(1, 1)) @ x @ cholesky(PhiPhiT_bar_inv)

            # Calculating the posterior pdf of kernel hyperparameters theta
            post_pdf = utils.mvnpdf_log(np.squeeze(utils.vectorize(A[dd][:, :, kk + 1])), np.squeeze(utils.vectorize(A_mean)), np.kron(PhiPhiT_bar_inv, Q[dd, dd, kk + 1, iter].reshape(1, 1))) \
                    + np.log(utils.iwishpdf(Q[dd, dd, kk + 1, iter].reshape(1, 1), Q_scale, T - 1 + lQ))

            # Metropolis_Hastings sampler for the kernel hyperparameters theta
            ls[dd][:, kk + 1] = ls[dd][:, kk]
            var[dd, kk + 1, iter] = var[dd, kk, iter]

            for k_mh in range(K_mh):
                # samples from proposal distribution (taken as Gaussian distribution)
                ls_prop = ls[dd][:, kk + 1] + 0.01 * normal(size=(len(dims[dd])))
                var_prop = var[dd, kk + 1, iter] + normal(size=1)

                if np.all(ls_prop > 0) and (var_prop > 0):
                    V_prop = V_prior(ls_prop, var_prop, dd)

                    PhiPhiT_bar_prop = PhiPhiT + V_prop
                    ZPhiT_bar_prop = ZPhiT
                    ZZT_bar_prop = ZZT

                    PhiPhiT_bar_inv_prop = pinv(PhiPhiT_bar_prop)
                    PhiPhiT_bar_inv_prop = 0.5 * (PhiPhiT_bar_inv_prop + PhiPhiT_bar_inv_prop.T)
                    A_mean_prop = ZPhiT_bar_prop @ PhiPhiT_bar_inv_prop
                    Q_scale_prop = LambdaQ + ZZT_bar_prop - A_mean @ ZPhiT_bar_prop.T
                    Q_scale_prop = 0.5 * (Q_scale_prop + Q_scale_prop.T)

                    # calculating the posterior pdf of the kernel hyperparameters using proposal distribution
                    post_pdf_prop = utils.mvnpdf_log(np.squeeze(utils.vectorize(A[dd][:, :, kk + 1])), np.squeeze(utils.vectorize(A_mean_prop)), np.kron(PhiPhiT_bar_inv_prop, Q[dd, dd, kk + 1, iter].reshape(1, 1))) \
                                + np.log(utils.iwishpdf(Q[dd, dd, kk + 1, iter].reshape(1, 1), Q_scale_prop, T - 1 + lQ))

                    # Accept-Reject condition
                    dv = uniform(size=1)
                    tmp = np.exp(post_pdf_prop + ls_prior(ls_prop, dd) + var_prior(var_prop) - post_pdf - ls_prior(ls[dd][:, kk + 1], dd) - var_prior(var[dd, kk + 1, iter]))
                    if np.isnan(tmp):
                        dl = 1
                    else:
                        dl = np.min((1, tmp))

                    # kernel hyperparameter update based on accept-reject condition
                    ls[dd][:, kk + 1] = (dv > dl) * ls[dd][:, kk + 1] + (dv <= dl) * ls_prop
                    var[dd, kk + 1, iter] = (dv > dl) * var[dd, kk + 1, iter] + (dv <= dl) * var_prop

        # updating the main parameters
        A0[:, :, kk + 1, iter] = A[0][:, :, kk + 1]
        A1[:, :, kk + 1, iter] = A[1][:, :, kk + 1]
        A2[:, :, kk + 1, iter] = A[2][:, :, kk + 1]
        ls0[:, kk + 1, iter] = ls[0][:, kk + 1]
        ls1[:, kk + 1, iter] = ls[1][:, kk + 1]
        ls2[:, kk + 1, iter] = ls[2][:, kk + 1]

'''
    Extracting final model
'''
# Remove burn-in
# Actual number of samples, if stopped prematurely
if kk < K:
    K = kk - 1

# Remove burn-in
burn_in = int(min(np.floor(K / 2), 100))
Ar = A[:, :, burn_in: K]
Qr = Q[:, :, burn_in: K]

# Show weight histograms
Kb = K - burn_in

# Extract learned model
covA = np.cov(np.squeeze(Ar))
meanA = np.mean(Ar, axis=2)

covQ = np.cov(np.squeeze(Qr))
meanQ = np.mean(Qr)

if SAVE:
    out_name = 'dagpssm_lv'
    output = pathlib.Path(out_name)
    results_path = output.with_suffix('.npz')
    with open(results_path, 'xb') as results_file:
        np.savez(results_file, x=Xt, y=Y, t=t, Q=Q, ls0=ls0, ls1=ls1, ls2=ls2, var=var, A0=A0, A1=A1, A2=A2, x_prim=x_prim, R=R, Lx=L, N=N, lQ=lQ, LambdaQ=LambdaQ, k_mh=k_mh, M=M, x_pf=x_pf, meanA=meanA, meanQ=meanQ)

print('finished!')