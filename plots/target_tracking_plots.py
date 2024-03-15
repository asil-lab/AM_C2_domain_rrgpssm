import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from numpy.linalg import norm
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import cholesky
mpl.use('Qt5Agg')
plt.ion()

tend = 40
dt = 0.1
t = np.arange(0, tend + dt, dt)
T = len(t)

data = np.load('robot2D_20022024.npz')
nx = 3
K = 1000
dims = [[0, 2, 3], [1, 2, 3], [2, 4]]

A0 = np.array([[-50.0, -30.0], [-50.0, 30.0], [0.0, -30.0], [0.0, 30.0], [50.0, -30.0], [50.0, 30.0]]).T
Xt =data['Xt']
Y =data['Y']

ls0 = data['ls0']
ls1 = data['ls1']
ls2 = data['ls2']

var = data['var']

x_prim = data['x_prim']

Ai0 = data['Ai0']
Ai1 = data['Ai1']
Ai2 = data['Ai2']

L_x = []
L_x.append(data['Lx0'])
L_x.append(data['Lx1'])
L_x.append(data['Lx2'])

Qi = data['Qi']
M = data['M']

# trajectory error norm
rmse = np.zeros((K + 1, nx))
for kk in range(K + 1):
    for jj in range(nx):
        rmse[kk, jj] = norm(Xt[:, jj] - x_prim[jj, :, kk])

# trajectory and errors
fig = plt.figure(tight_layout=True)
gs = GridSpec(3, 1, figure=fig)
ax0 = fig.add_subplot(gs[:2, 0])
ax1 = fig.add_subplot(gs[2, 0])

ax0.plot(Xt[:, 0], Xt[:, 1], 'b', lw=3., label='ground truth')
ax0.plot(x_prim[0, :, -1], x_prim[1, :, -1], 'r.', label='estimate')
ax0.plot(A0[0, :], A0[1, :], 'kd', label='anchors')
ax0.legend()
ax0.set_xlabel('x [m]')
ax0.set_ylabel('y [m]')
ax0.grid()

ax1.set_yscale('log')
ax1.plot(np.arange(K + 1), rmse[:, 0], 'b-', label='x')
ax1.plot(np.arange(K + 1), rmse[:, 1], 'r-', label='y')
# ax11.plot(np.arange(K + 1), rmse[:, 2], 'b.-')
ax1.set_ylabel('rmse [m]')
ax1.set_xlabel('iterations [-]')
ax1.legend()
ax1.grid()
fig.suptitle('Trajectory estimation for mobile robot')

# lengthscale and variance
fig2, axs2 = plt.subplots(5, 1)
axs2[0].plot(np.arange(K + 1), ls0[0, :], 'b.')
axs2[0].text(0.2, 0.5, 'state x')
axs2[0].grid()

axs2[1].text(0.4, 0.5, 'state y')
axs2[1].plot(np.arange(K + 1), ls1[0, :], 'r.-')
axs2[1].grid()

axs2[2].text(0.5, 0.5, r'state $\theta$')
axs2[2].plot(np.arange(K + 1), ls0[1, :], 'b.-')
axs2[2].plot(np.arange(K + 1), ls1[1, :], 'r.-')
axs2[2].plot(np.arange(K + 1), ls2[0, :], 'k.-')
axs2[2].grid()

axs2[3].text(0.75, 0.5, 'input v')
axs2[3].plot(np.arange(K + 1), ls0[2, :], 'b.-')
axs2[3].plot(np.arange(K + 1), ls1[2, :], 'r.-')
axs2[3].grid()

axs2[4].text(0.9, 0.5, r'input \omega')
axs2[4].plot(np.arange(K + 1), ls1[2, :], 'r.-')
axs2[4].grid()

plt.figure()
plt.plot(np.arange(K + 1), var[0, :], 'b.-', label='x')
plt.plot(np.arange(K + 1), var[1, :], 'r.-', label='y')
# plt.plot(np.arange(K + 1), var[2, :], 'k.-', label=r'$\theta$')
plt.legend()
plt.grid()

# ax01.plot(np.arange(start, kk + 1), ell_f[1, start:kk + 1], 'rs-')
# ax01.plot(np.arange(start, kk + 1), ell_f[2, start:kk + 1], 'rs-')
# ax01.plot(np.arange(start, kk + 1), Sf_f[0, start:kk + 1], 'kd-')
# ax01.grid()

burn_in = int(min(np.floor(K / 2), 100))
Ar = [None] * nx
Ar[0] = Ai0[:, :, burn_in: K]
Ar[1] = Ai1[:, :, burn_in: K]
Ar[2] = Ai2[:, :, burn_in: K]
Qr = Qi[:, :, burn_in: K]
Kb = K - burn_in

# Extract learned model
covA = [None] * nx
meanA = [None] * nx
for nn in range(nx):
    covA[nn] = np.cov(np.squeeze(Ar[nn]))
    meanA[nn] = np.mean(Ar[nn], axis=2)
# covQ = np.cov(np.squeeze(Qr))
meanQ = np.mean(Qr, axis=2)

# Prediction error
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

def kinematic_model(x, u, dt=0.1):
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[2] += u[1] * dt

    return x

def control_inputs(delt, nT, traj):
    delx = np.diff(traj[0, :])
    dely = np.diff(traj[1, :])
    vx = delx / delt
    vy = dely / delt
    v = np.sqrt(vx ** 2 + vy ** 2)

    theta = np.zeros(nT)
    theta[1:] = np.arctan2(dely, delx)
    omega = np.diff(theta) / delt

    return np.hstack((v.reshape(nT - 1, 1), omega.reshape(nT - 1, 1)))

phi_x = lambda x, idx: np.prod(np.multiply(np.power(L_x[idx], -1 / 2), np.sin(np.pi * np.multiply(np.multiply(jv_x[idx], (np.reshape(x, (1, 1, L_x[idx].shape[2])) + L_x[idx])), 1. / (2 * L_x[idx])))), axis=2)

jv_x = []
lambda_x = []
for ii in range(nx):
    jv, eigvals = se_kernel_eigvals(M, L_x[ii])
    jv_x.append(jv)
    lambda_x.append(eigvals)

tend = 3
dt = 0.1
t = np.arange(0, tend + dt, dt)
T = len(t)
test_traj = np.zeros((T, 2))
test_traj[:, 0] = np.linspace(40, 50, T)
test_traj[:, 1] = 20 + np.cos(test_traj[:, 0] - 40)
U = control_inputs(dt, T, test_traj.T)

# Simulate prediction trajectory
Xt = np.zeros((T, nx))
Xt[0, :-1] = np.copy(test_traj[0, :])
fmean = np.zeros((T, nx))
fstd = np.zeros((T, nx))
f_last = np.zeros((T, nx))
for tt in range(1, len(t)):
    Xt[tt, :] = kinematic_model(Xt[tt - 1, :], U[tt - 1, :])
    for nn in range(nx):
        x_tmp = np.concatenate((Xt[tt-1, :], U[tt-1, :]))
        phi_tmp = phi_x(x_tmp[dims[nn]], nn)
        f_last[tt, nn] = Ar[nn][:, :, -1] @ phi_tmp + np.sqrt(meanQ[nn, nn]) * normal(size=1)
        fmean[tt, nn] = meanA[nn] @ phi_tmp + np.sqrt(meanQ[nn, nn]) * normal(size=1)
        fstd[tt, nn] = np.sqrt(np.diag(phi_tmp.reshape(1, M ** len(dims[nn])) @ covA[nn] @ phi_tmp.reshape(M ** len(dims[nn]), 1)))

fig4, axs4 = plt.subplots(nx, 1)
# Uncertainty
# axs4.fill_between([xv, flip(xv)], [f_m(xv) + 2 * f_std(xv), flip(f_m(xv) - 2 * f_std(xv))], 0.95 * [1 1 1], 'EdgeColor', 'none')
# Show true
for nn in range(nx):
    # Show uncertainty
    axs4[nn].plot(t, Xt[:, nn], 'k', linewidth=2)
    axs4[nn].plot(t, fmean[:, nn], 'b', linewidth=2)
    axs4[nn].plot(t, fmean[:, nn] + 2 * fstd[:, nn], 'grey', linewidth=1)
    axs4[nn].plot(t, fmean[:, nn] - 2 * fstd[:, nn], 'grey', linewidth=1)
    axs4[nn].plot(t, f_last[:, nn], 'r', linewidth=2)

axs4[0].set_ylabel(r'$x_t$')
axs4[1].set_ylabel(r'$y_t$')
axs4[2].set_ylabel(r'$\theta_t$')
axs4[2].set_xlabel('t [s]')

# axs4.set_legend('Posterior mean', '2\sigma of posterior', 'True function', 'location', 'northwest')
# axs4.set_xlim([np.min(xv), np.max(xv)])
# axs4.set_ylim([-3, 3])
plt.show()

print('finished!')