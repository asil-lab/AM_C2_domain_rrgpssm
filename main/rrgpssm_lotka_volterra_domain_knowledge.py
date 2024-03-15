import numpy as np
import utils
from numpy.random import normal, uniform
from numpy.linalg import cholesky, pinv, norm
import scipy.stats as sps
from scipy.stats import multivariate_normal as mvn
from numpy.matlib import repmat
import pathlib

NOISE = True
save_flag = True
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

def se_kernel_eigvals(M, L):
    # 3D (ANU: range changed to match matlab code)
    d = L.shape[2]
    jv = np.zeros((M ** d, 1, d))
    eigvals = np.zeros((M ** d, d))
    if d == 3:
        for ii in range(1, M + 1):
            for jj in range(1, M + 1):
                for kk in range(1, M + 1):
                    ind = M ** 2 * (ii - 1) + M * (jj - 1) + kk
                    jv[ind - 1, 0, :] = np.array([ii, jj, kk])
                    eigvals[ind - 1, :] = np.power(np.pi * jv[ind - 1, 0, :] / (2. * np.squeeze(L)), 2)
    elif d == 2:
        for jj in range(1, M + 1):
            for kk in range(1, M + 1):
                ind = M * (jj - 1) + kk
                jv[ind - 1, 0, :] = np.array([jj, kk])
                eigvals[ind - 1, :] = np.power(np.pi * jv[ind - 1, 0, :] / (2. * np.squeeze(L)), 2)
    else:
        raise Exception("This dimensionality has not been implemented!")

    return jv, eigvals

# Set random number generator(for reproducible results)
SEED = 1
np.random.seed(SEED)

# Measurement function
g = lambda x: x

tend = 40
dt = 0.1
t = np.arange(0, tend + dt, dt)
T = len(t)
nx = 3

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
    Y[tt, :] = g(X[tt, :]) + mvn.rvs(mean=np.zeros(nx), cov=R)

Y[0, :] = g(X[0, :]) + mvn.rvs(mean=np.zeros(nx), cov=R)

# Parameters for the algorithm, priors, and basis functions
K = 100
N = 5
K_mh = 5

# Priors for Q
lQ = 1
LambdaQ = 1 * np.identity(nx)

# Basis functions for f:
M = 6
M_bar = [M ** 2, M ** 2, M ** 3]

# [0, 1, 2, 3, 4] = [x, y, theta, v, omega]
dims = [[0, 1], [1, 2], [0, 1, 2]]

# Boundary conditions
L_x = [np.array([[[5., 5.]]]), np.array([[[5., 5.]]]), np.array([[[5., 5., 5.]]])]

# Eigenvalue/spectral density
jv_x = []
lambda_x = []
for ii in range(nx):
    jv, eigvals = se_kernel_eigvals(M, L_x[ii])
    jv_x.append(jv)
    lambda_x.append(eigvals)

# Eigenfunctions
phi_x = lambda x, idx: np.prod(np.multiply(np.power(L_x[idx], -1 / 2), np.sin(np.pi * np.multiply(np.multiply(jv_x[idx], (np.reshape(x, (1, 1, L_x[idx].shape[2])) + L_x[idx])), 1. / (2 * L_x[idx])))), axis=2)

# GP prior:
S_SE = lambda w, ell: np.sqrt(2 * np.pi * ell ** 2) * np.exp(-(w ** 2 / 2.) * ell ** 2)
S_f = lambda lx, Sfx, idx: 1 / Sfx * np.diag(np.prod(1. / S_SE(np.sqrt(lambda_x[idx]), np.tile(lx, (M_bar[idx], 1))), 1))

# Priors for ell
ell_f_prior = lambda ell, idx: utils.mvnpdf_log(np.log(ell), np.squeeze(repmat(np.log(10), len(dims[idx]), 1)), np.identity(len(dims[idx])))
Sf_f_prior = lambda Sf_f: utils.mvnpdf_log(Sf_f, 100, 100)

# Pre-allocate and initialization
Qi = np.zeros((nx, nx, K+1, NEXP))
Ai0 = np.zeros((1, M_bar[0], K + 1, NEXP))
Ai1 = np.zeros((1, M_bar[1], K + 1, NEXP))
Ai2 = np.zeros((1, M_bar[2], K + 1, NEXP))
ell_f0 = np.zeros((len(dims[0]), K + 1, NEXP))
ell_f1 = np.zeros((len(dims[1]), K + 1, NEXP))
ell_f2 = np.zeros((len(dims[2]), K + 1, NEXP))
Sf_f = np.zeros((nx, K + 1, NEXP))
x_prim = np.zeros((nx, T, K + 1, NEXP))
x_pf = np.zeros((nx, N, T, K, NEXP))
err_norm = np.zeros((K + 1, nx, NEXP))

for iter in range(NEXP):
	SEED = 1 + iter
	np.random.seed(SEED)
	Ai = []
	ell_f = []
	for nn in range(nx):
		Ai.append(np.zeros((1, M_bar[nn], K + 1)))
		Ai[nn][0, :, 0] = 10 * normal(size=M_bar[nn])
		ell_f.append(np.zeros((len(dims[nn]), K + 1)))
		ell_f[nn][:, 0] = uniform(size=len(dims[nn]))
	Qi[:, :, 0, iter] = 1 * np.identity(nx)
	Sf_f[:, 0, iter] = 1 * np.ones(nx)
	
	for jj in range(nx):
		err_norm[0, jj, iter] = norm(Xt[:, jj] - x_prim[jj, :, 0, iter])
	
	# Run MCMC algorithm!
	for kk in range(K):
		f_i = lambda x, idx: Ai[idx][:, :, kk] @ phi_x(x, idx).reshape(M_bar[idx], 1)
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
		    # PF time propagation, resampling and ancestor sampling
				
			if tt >= 1:
				a[tt, :-1] = utils.systematic_resampling(w[tt - 1, :], N - 1)
				for jj in range(N - 1):
					for nn in range(nx):
						x_bar = x_pf[:, int(a[tt, jj]), tt - 1, kk, iter]
						x_pf[nn, jj, tt, kk, iter] = np.squeeze(f_i(x_bar[dims[nn]], nn) + Q_chol[nn, nn] * normal(size=1))
						
				waN = np.zeros(N)
				xN = np.zeros(nx) # initiliazing the last particle
				for jj in range(N):
					for nn in range(nx):
						x_bar = x_pf[:, jj, tt - 1, kk, iter]
						xN[nn] = np.squeeze(f_i(x_bar[dims[nn]], nn))
					waN[jj] = w[tt - 1, jj] * mvn.pdf(x_pf[:, -1, tt, kk, iter], mean=xN, cov=Qi[:, :, kk, iter])
		            
				waN = waN / np.sum(waN)
				a[tt, -1] = utils.systematic_resampling(waN, 1)
			
			# PF weight update
			for jj in range(N):
				w[tt, jj] = mvn.pdf(x_pf[:, jj, tt, kk, iter], mean=Y[tt, :], cov=R)
				
			# w[tt, :] = np.exp(log_w - np.max(log_w))
			w[tt, :] = w[tt, :] / np.sum(w[tt, :])

		# Sample trajectory to condition on
		star = utils.systematic_resampling(w[-1, :], 1)
		x_prim[:, -1, kk + 1, iter] = x_pf[:, star, -1, kk, iter]
		for tt in np.arange(T-2, -1,  -1):
			star = a[tt + 1, int(star)]
			x_prim[:, tt, kk + 1, iter] = x_pf[:, int(star), tt, kk, iter]

		print('Sampling. k = ' + str(kk + 1) + '/' + str(K) + ' of Exp: ' + str(iter + 1) + '/' + str(NEXP))
		
		# ANU: Dealing with each output(state-space) dimension separately
		for nn in range(nx):
		    # Compute statistics
			xiX = np.expand_dims(x_prim[nn, 1: T, kk + 1, iter].reshape(1, T - 1), axis=1)
			
			zX = np.zeros((M_bar[nn], 1, T - 1))
			for tt in range(T - 1):
				x_bar = x_prim[:, tt, kk + 1, iter]
				zX[:, 0, tt] = np.squeeze(phi_x(x_bar[dims[nn]], nn))
				
			tmp = np.asarray(np.tile(xiX, (1, 1, 1)))
			PhiX = np.sum(tmp * xiX.swapaxes(0, 1), axis=2)
			
			tmp = np.asarray(np.tile(xiX, (1, M_bar[nn], 1)))
			PsiX = np.sum(tmp * zX.swapaxes(0, 1), axis=2)
			
			tmp = np.asarray(np.tile(zX, (1, M_bar[nn], 1)))
			SigmaX = np.sum(tmp * zX.swapaxes(0, 1), axis=2)

		    # Sample new parameters
			VX = S_f(ell_f[nn][:, kk], Sf_f[nn, kk, iter], nn)
			
			SigmaX_bar = SigmaX + VX
			PsiX_bar = PsiX
			PhiX_bar = PhiX
			
			SigmaX_bar_inv = pinv(SigmaX_bar)
			GammaX_star = PsiX_bar @ SigmaX_bar_inv
			PiX_star = PhiX_bar - GammaX_star @ PsiX_bar.T
			
			PiX_star_fix = 0.5 * (PiX_star + PiX_star.T)
			SigmaX_bar_inv_fix = 0.5 * (SigmaX_bar_inv + SigmaX_bar_inv.T)
			
			Qi[nn, nn, kk + 1, iter] = sps.invwishart.rvs(df=T - 1 + lQ, scale=LambdaQ[nn, nn] + PiX_star_fix)
			x = normal(size=(1, M_bar[nn]))
			if np.all(np.real(np.linalg.eigvals(SigmaX_bar_inv_fix)) > 0):
				GamX = GammaX_star + cholesky(Qi[nn, nn, kk + 1, iter].reshape(1, 1)) @ x @ cholesky(SigmaX_bar_inv_fix)
			else:
				lmd = np.floor(np.log10(np.abs(np.min(np.real(np.linalg.eigvals(SigmaX_bar_inv_fix))))))
				GamX = GammaX_star + cholesky(Qi[nn, nn, kk + 1, iter].reshape(1, 1)) @ x @ cholesky(SigmaX_bar_inv_fix + 10 ** (lmd + 1) * np.eye(M_bar[nn]))
			Ai[nn][:, :, kk + 1] = GamX

		    # Likelihood of S
			p_S_f = utils.mvnpdf_log(np.squeeze(utils.vectorize(GamX)), np.squeeze(utils.vectorize(GammaX_star)), np.kron(SigmaX_bar_inv_fix, Qi[nn, nn, kk + 1, iter].reshape(1, 1))) \
		            + np.log(utils.iwishpdf(Qi[nn, nn, kk + 1, iter].reshape(1, 1), LambdaQ[nn, nn] + PiX_star_fix, T - 1 + lQ))
			
			ell_f[nn][:, kk + 1] = ell_f[nn][:, kk]
			Sf_f[nn, kk + 1, iter] = Sf_f[nn, kk, iter]

			# Run MH sampler
			for k_mh in range(K_mh):
				ell_prop = ell_f[nn][:, kk + 1] + 0.01 * normal(size=(len(dims[nn])))
				Sf_prop = Sf_f[nn, kk + 1, iter] + normal(size=1)
				
				if np.all(ell_prop > 0) and (Sf_prop > 0):
					V_prop = S_f(ell_prop, Sf_prop, nn)
					
					SigmaX_bar_prop = SigmaX + V_prop
					PsiX_bar_prop = PsiX # +M * V
					PhiX_bar_prop = PhiX # +M * V * M'
					
					SigmaX_bar_inv_prop = pinv(SigmaX_bar_prop)
					GammaX_star_prop = PsiX_bar_prop @ SigmaX_bar_inv_prop
					PiX_star_prop = PhiX_bar_prop - GammaX_star_prop @ PsiX_bar_prop.T
					
					SigmaX_bar_inv_prop_fix = 0.5 * (SigmaX_bar_inv_prop + SigmaX_bar_inv_prop.T)
					
					p_S_prop = utils.mvnpdf_log(np.squeeze(utils.vectorize(GamX)), np.squeeze(utils.vectorize(GammaX_star)), np.kron(SigmaX_bar_inv_prop_fix, Qi[nn, nn, kk + 1, iter].reshape(1, 1))) \
		                        + np.log(utils.iwishpdf(Qi[nn, nn, kk + 1, iter].reshape(1, 1), LambdaQ[nn, nn] + PiX_star_prop, T - 1 + lQ))
					
					dv = uniform(size=1)
					tmp = np.exp(p_S_prop + ell_f_prior(ell_prop, nn) + Sf_f_prior(Sf_prop) - p_S_f - ell_f_prior(ell_f[nn][:, kk + 1], nn) - Sf_f_prior(Sf_f[nn, kk + 1, iter]))
					if np.isnan(tmp):
						dl = 1
					else:
						dl = np.min((1, tmp))
					
					ell_f[nn][:, kk + 1] = (dv > dl) * ell_f[nn][:, kk + 1] + (dv <= dl) * ell_prop
					Sf_f[nn, kk + 1, iter] = (dv > dl) * Sf_f[nn, kk + 1, iter] + (dv <= dl) * Sf_prop
		
		# updating the remaining parameters
		Ai0[:, :, kk + 1, iter] = Ai[0][:, :, kk + 1]
		Ai1[:, :, kk + 1, iter] = Ai[1][:, :, kk + 1]
		Ai2[:, :, kk + 1, iter] = Ai[2][:, :, kk + 1]
		ell_f0[:, kk + 1, iter] = ell_f[0][:, kk + 1]
		ell_f1[:, kk + 1, iter] = ell_f[1][:, kk + 1]
		ell_f2[:, kk + 1, iter] = ell_f[2][:, kk + 1]

# Extract and evaluate model

if save_flag:
    out_name = 'rrgpssm_lv_dom_MC_14032024'
    output = pathlib.Path(out_name)
    results_path = output.with_suffix('.npz')
    with open(results_path, 'xb') as results_file:
        np.savez(results_file, xt=Xt, x=X, y=Y, t=t, Q=Q, ls0=ell_f0, ls1=ell_f1, ls2=ell_f2, var=Sf_f, Qi=Qi, Ai0=Ai0, Ai1=Ai1, Ai2=Ai2, x_prim=x_prim, R=R, Lx0=L_x[0], Lx1=L_x[1], Lx2=L_x[2], N=N, lQ=lQ, LambdaQ=LambdaQ, k_mh=k_mh, M=M, x_pf=x_pf)

print('finished!')
