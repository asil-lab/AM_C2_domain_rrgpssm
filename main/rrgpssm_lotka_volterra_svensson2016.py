import numpy as np
import utils
from numpy.random import normal, uniform
from numpy.linalg import cholesky, pinv, norm, det
import scipy.stats as sps
from scipy.stats import multivariate_normal as mvn
from numpy.matlib import repmat
import pathlib
NOISE = True
NEXP = 100

# Define model and simulate data
def lv_dyn(x, alpha=np.array([3, 4, 7.2]), beta=np.array([[-0.5, -1, 0], [0, -1, -2], [-2.6, -1.6, -3]])):
    L = len(x)
    Dx = np.zeros(L)
    for ii in range(L):
        Dx[ii] = alpha[ii] * x[ii]
        for jj in range(L):
            Dx[ii] += beta[ii, jj] * x[ii] * x[jj]

    return Dx

# Set random number generator(for reproducible results)
SEED = 1
np.random.seed(SEED)
save_flag = True

# Define the true functions g
g = lambda x: x

tend = 40
dt = 0.1
t = np.arange(0, tend + dt, dt)
T = len(t)

nx = 3
# nx_bar = int((nx * (nx + 1.)) / 2.)
idx = np.triu_indices(nx)
# Simulate trajectory
Xt = np.zeros((len(t), nx))
X = np.zeros((len(t), nx))
Y = np.zeros((len(t), nx))
Q = 0.01 * np.eye(nx)
R = 0.01 * np.eye(nx)
X[0, :] = np.array([0.1, 0.8, 0.3])
Xt[0, :] = np.array([0.1, 0.8, 0.3])

for tt in range(1, len(t)):
    # x[tt + 1] = f(x[tt]) + mvnrnd(0, Q)
    Xt[tt, :] = Xt[tt - 1, :] + dt * lv_dyn(Xt[tt - 1, :])
    X[tt, :] = Xt[tt - 1, :] + dt * lv_dyn(X[tt - 1, :]) + NOISE * mvn.rvs(mean=np.zeros(nx), cov=Q)
    Y[tt, :] = g(X[tt, :]) + NOISE * mvn.rvs(mean=np.zeros(nx), cov=R)
Y[0, :] = g(X[0, :]) + NOISE * mvn.rvs(mean=np.zeros(nx), cov=R)

# Parameters for the algorithm, priors, and basis functions
K = 100
N = 5
K_mh = 5

# Priors for Q
alpha = 1
Beta = 1 * np.identity(nx)

# Basis functions for f:
M = 6
M_bar = M ** nx

# Boundary conditions
L_x = 5 * np.ones((1, 1, nx))

# Eigenvalue/spectral density
jv_x = np.zeros((M_bar, 1, nx))
lambda_x = np.zeros((M_bar, nx))
for ii in range(1, M + 1):
    for jj in range(1, M + 1):
        for kk in range(1, M + 1):
            ind = M ** 2 * (ii - 1) + M * (jj - 1) + kk
            jv_x[ind - 1, 0, :] = np.array([ii, jj, kk])
            lambda_x[ind - 1, :] = np.power(np.pi * jv_x[ind - 1, :] / (2. * np.squeeze(L_x)), 2)

# Eigenfunctions
phi_x = lambda x: np.prod(np.multiply(np.power(L_x, -1 / 2), np.sin(np.pi * np.multiply(np.multiply(jv_x, (np.reshape(x, (1, 1, nx)) + L_x)), 1. / (2 * L_x)))), axis=2)

# GP prior:
S_SE = lambda w, ell: np.sqrt(2 * np.pi * ell ** 2) * np.exp(-(w ** 2 / 2.) * ell ** 2)
S_f = lambda lx, Sfx: 1 / Sfx * np.diag(np.prod(1. / S_SE(np.sqrt(lambda_x), np.tile(lx, (M_bar, 1))), 1))

# Priors for lengthscale and variance
ell_f_prior = lambda ell: utils.mvnpdf_log(np.log(ell), np.squeeze(repmat(np.log(10), nx, 1)), np.identity(nx))
Sf_f_prior = lambda Sf_f: utils.mvnpdf_log(Sf_f, 100, 100)

# Pre-allocate and initialization
Qi = np.zeros((nx, nx, K+1, NEXP))
Ai = np.zeros((nx, M_bar, K + 1, NEXP))
ell_f = np.zeros((nx, K + 1, NEXP))
Sf_f = np.zeros((1, K + 1, NEXP))

x_pf = np.zeros((nx, N, T, K, NEXP))	
x_prim = np.zeros((nx, T, K + 1, NEXP))
err_norm = np.zeros((K + 1, nx, NEXP))

for iter in range(NEXP):
	SEED = 1 + iter
	np.random.seed(SEED)

	Qi[:, :, 0, iter] = 1 * np.identity(nx)
	Ai[:, :, 0, iter] = 10 * normal(size=(nx, M_bar))
	ell_f[:, 0, iter] = uniform(size=(nx))
	Sf_f[0, 0, iter] = 1
	
	for jj in range(nx):
    		err_norm[0, jj, iter] = norm(Xt[:, jj] - x_prim[jj, :, 0, iter])

	# Run MCMC algorithm!
	for kk in range(K):
		f_i = lambda x: Ai[:, :, kk, iter] @ phi_x(x).reshape(M_bar, 1)
		Q_chol = cholesky(Qi[:, :, kk, iter])

		# Pre - allocate
		w = np.zeros((T, N))
		a = np.zeros((T, N))

		# Initialize
		if kk > 0:
		    x_pf[:, -1, :, kk, iter] = x_prim[:, :, kk, iter]

		w[0, :] = 1
		w[0, :] = w[0, :] / np.sum(w[0, :])

		# CPF with ancestor sampling
		x_pf[:, 0: -1, 0, kk, iter] = np.tile(Y[0, :].reshape(nx, 1), (1, N - 1))
		x_pf[:, -1, 0, kk, iter] = x_prim[:, 0, kk, iter]

		for tt in range(T):
		    print([kk, tt])
		    # PF time propagation, resampling and ancestor sampling
		    if tt >= 1:
		        a[tt, :-1] = utils.systematic_resampling(w[tt - 1, :], N - 1)
		        for jj in range(N - 1):
		            x_pf[:, jj, tt, kk, iter] = np.squeeze(f_i(x_pf[:, int(a[tt, jj]), tt - 1, kk, iter]) + Q_chol @ normal(size=(nx, 1)))

		        # x_pf[:, -1, tt] = x_prim[:, tt, kk]
		        waN = np.zeros(N)
		        for jj in range(N):
		        	waN[jj] = w[tt - 1, jj] * mvn.pdf(x_pf[:, -1, tt, kk, iter], mean=np.squeeze(f_i(x_pf[:, jj, tt - 1, kk, iter])), cov=Qi[:, :, kk, iter])
		            # waN[jj] = w[tt - 1, jj] * mvn.pdf(np.squeeze(f_i(x_pf[:, jj, tt - 1])), x_pf[:, -1, tt], Qi[:, :, kk])
		        waN = waN / np.sum(waN)
		        a[tt, -1] = utils.systematic_resampling(waN, 1)

		    # PF weight update
		    # log_w = np.zeros(N)
		    for jj in range(N):
		        w[tt, jj] = mvn.pdf(x_pf[:, jj, tt, kk, iter], mean=Y[tt, :], cov=R)
		        # log_w[jj] = -np.power(g(x_pf[:, jj, tt]) - Y[tt, :], 2) / 2 @ pinv(R)

		    # w[tt, :] = np.exp(log_w - np.max(log_w))
		    w[tt, :] = w[tt, :] / np.sum(w[tt, :])

		# Sample trajectory to condition on
		star = utils.systematic_resampling(w[-1, :], 1)
		x_prim[:, -1, kk + 1, iter] = x_pf[:, star, -1, kk, iter]
		for tt in np.arange(T-2, -1,  -1):
		    star = a[tt + 1, int(star)]
		    x_prim[:, tt, kk + 1, iter] = x_pf[:, int(star), tt, kk, iter]

		print('Sampling. k = ' + str(kk + 1) + '/' + str(K) + ' of Exp: ' + str(iter + 1) + '/' + str(NEXP))
		
		# Compute statistics from Wills2012
		xiX = np.expand_dims(x_prim[:, 1: T, kk + 1, iter], axis=1)

		zX = np.zeros((M_bar, 1, T - 1))
		for tt in range(T - 1):
		    zX[:, 0, tt] = np.squeeze(phi_x(x_prim[:, tt, kk + 1, iter]))
		
		tmp = np.asarray(np.tile(xiX, (1, nx, 1)))
		PhiX = np.sum(tmp * xiX.swapaxes(0, 1), axis=2)

		tmp = np.asarray(np.tile(xiX, (1, M_bar, 1)))
		PsiX = np.sum(tmp * zX.swapaxes(0, 1), axis=2)

		tmp = np.asarray(np.tile(zX, (1, M_bar, 1)))
		SigmaX = np.sum(tmp * zX.swapaxes(0, 1), axis=2)

		# Sample new parameters
		VX = S_f(ell_f[:, kk, iter], Sf_f[0, kk, iter])

		SigmaX_bar = SigmaX + VX
		PsiX_bar = PsiX
		PhiX_bar = PhiX

		SigmaX_bar_inv = pinv(SigmaX_bar)
		GammaX_star = PsiX_bar @ SigmaX_bar_inv
		PiX_star = PhiX_bar - GammaX_star @ PsiX_bar.T

		PiX_star_fix = 0.5 * (PiX_star + PiX_star.T)
		SigmaX_bar_inv_fix = 0.5 * (SigmaX_bar_inv + SigmaX_bar_inv.T)

		Qi[:, :, kk + 1, iter] = sps.invwishart.rvs(df=T - 1 + lQ, scale=LambdaQ + PiX_star_fix)
		x = normal(size=(nx, M_bar))
		GamX = GammaX_star + cholesky(Qi[:, :, kk + 1, iter]) @ x @ cholesky(SigmaX_bar_inv_fix)
		Ai[:, :, kk + 1, iter] = GamX

		# Likelihood
		p_S_f = utils.mvnpdf_log(np.squeeze(utils.vectorize(GamX)), np.squeeze(utils.vectorize(GammaX_star)), np.kron(SigmaX_bar_inv_fix, Qi[:, :, kk + 1, iter])) \
		        + np.log(utils.iwishpdf(Qi[:, :, kk + 1, iter], LambdaQ + PiX_star_fix, T - 1 + lQ))

		ell_f[:, kk + 1, iter] = ell_f[:, kk, iter]
		Sf_f[0, kk + 1, iter] = Sf_f[0, kk, iter]

		# Run MH sampler
		for k_mh in range(K_mh):
		    ell_prop = ell_f[:, kk + 1, iter] + 0.01 * normal(size=(nx))
		    Sf_prop = Sf_f[0, kk + 1, iter] + normal(size=1)

		    if np.all(ell_prop > 0) and (Sf_prop > 0):
		        V_prop = S_f(ell_prop, Sf_prop)

		        SigmaX_bar_prop = SigmaX + V_prop
		        PsiX_bar_prop = PsiX # +M * V
		        PhiX_bar_prop = PhiX # +M * V * M'

		        SigmaX_bar_inv_prop = pinv(SigmaX_bar_prop)
		        GammaX_star_prop = PsiX_bar_prop @ SigmaX_bar_inv_prop
		        PiX_star_prop = PhiX_bar_prop - GammaX_star_prop @ PsiX_bar_prop.T

		        SigmaX_bar_inv_prop_fix = 0.5 * (SigmaX_bar_inv_prop + SigmaX_bar_inv_prop.T)

		        p_S_prop = utils.mvnpdf_log(np.squeeze(utils.vectorize(GamX)), np.squeeze(utils.vectorize(GammaX_star)), np.kron(SigmaX_bar_inv_prop_fix, Qi[:, :, kk + 1, iter])) \
		                    + np.log(utils.iwishpdf(Qi[:, :, kk + 1, iter], LambdaQ + PiX_star_prop, T - 1 + lQ))

		        dv = uniform(size=1)
		        tmp = np.exp(p_S_prop + ell_f_prior(ell_prop) + Sf_f_prior(Sf_prop) - p_S_f - ell_f_prior(ell_f[:, kk + 1, iter]) - Sf_f_prior(Sf_f[:, kk + 1, iter]))
		        if np.isnan(tmp):
		            dl = 1
		        else:
		            dl = np.min((1, tmp))

		        ell_f[:, kk + 1, iter] = (dv > dl) * ell_f[:, kk + 1, iter] + (dv <= dl) * ell_prop
		        Sf_f[:, kk + 1, iter] = (dv > dl) * Sf_f[:, kk + 1, iter] + (dv <= dl) * Sf_prop

if save_flag:
    out_name = 'rrgpssm_lv_MC_' + str(NEXP)
    output = pathlib.Path(out_name)
    results_path = output.with_suffix('.npz')
    with open(results_path, 'xb') as results_file:
        np.savez(results_file, x=Xt, y=Y, Q=Q, t=t, ls=ell_f, var=Sf_f, Qi=Qi, Ai=Ai, x_prim=x_prim, R=R, Lx=L_x, N=N, lQ=lQ, LambdaQ=LambdaQ, k_mh=k_mh, M=M, x_pf=x_pf)

print('finished!')