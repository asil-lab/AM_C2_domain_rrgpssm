import numpy as np
from numpy.random import seed, normal, uniform, randn, permutation
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import det, norm, pinv, inv, eig
from scipy.special import loggamma
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import reduce
from math import copysign, hypot
mpl.use('Qt5Agg')
plt.ion()

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def edm(X):
    d, n = X.shape

    one = np.ones((n, 1))
    G = X.transpose().dot(X)
    g = G.diagonal().reshape((n, 1))
    D = g.dot(one.transpose()) + one.dot(g.transpose()) - 2.0 * G

    return D

def systematic_resampling(W, N):
    # Systematic resampling
    W = W / np.sum(W)
    u = 1/N * uniform(size=1)

    idx = np.zeros((N, 1))
    q = 0
    # ANU: to take care of index difference in MATLAB and python
    n = -1
    for ii in range(N):
        while q < u:
            n = n + 1
            q = q + W[n]

        idx[ii] = n
        u = u + 1 / N

    return idx.astype(int).squeeze()

def mvnpdf_log(X, Mu, Sigma):
    if not np.isscalar(Sigma) and not np.all(np.real(np.linalg.eigvals(Sigma)) > 0):
        lmd = np.floor(np.log10(np.abs(np.min(np.real(np.linalg.eigvals(Sigma))))))
        Sigma = Sigma + 10 ** (lmd + 1) * np.eye(Sigma.shape[0])
    return np.log(mvn.pdf(X, Mu, Sigma, allow_singular=True))

def iwishpdf(W, Tau, df):
    p, _ = W.shape

    # TODO: Check how the dimension of p  affects the step below as later only 1 value of gamma_ln is expected
    for pp in range(p):
        gamma_ln = loggamma(df / 2. + pp / 2.)

    gamma_np = np.log(np.pi) * ((p - 1) * p / 4) + np.sum(gamma_ln)
    num = np.log(det(W)) * (-(p + df + 1) / 2) + np.log(det(Tau)) * (df / 2) - 0.5 * np.trace(Tau / W)
    den = np.log(2) * (p * df / 2) + gamma_np
    logpdf = num - den
    Y = np.exp(logpdf)

    return Y

def vectorize(A, ch=False):
    rows, cols = A.shape
    if ch:
        vecA = np.chararray((rows * cols, 1))
    else:
        vecA = np.zeros((rows * cols, 1))

    for ii in range(cols):
        # vecA[ii * cols: (ii + 1) * cols] = A[:, ii].reshape((rows, 1))
        vecA[ii * rows: (ii + 1) * rows] = A[:, ii].reshape((rows, 1))

    return vecA