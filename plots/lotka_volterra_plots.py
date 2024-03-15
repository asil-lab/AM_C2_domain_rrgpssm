import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from numpy.linalg import norm
mpl.use('Qt5Agg')
font = {'size'   : 12}
mpl.rc('font', **font)

plt.ion()

def se_kernel_eigvals(M, L):
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

lv1 = np.load('rrgpssm_lv_MC_13032024.npz')
lv2 = np.load('rrgpssm_lv_dom_MC_14032024.npz')

nx = 3
K = 100
NEXP = 100

t = lv1['t']
T = len(t)
X =lv1['x']
Y =lv1['y']
L_x1 = lv1['Lx']
L_x2 = [lv2['Lx0'], lv2['Lx1'], lv2['Lx2']]
M1 = lv1['M']
M2 = lv2['M']
dims = [[0, 1], [1, 2], [0, 1, 2]]
start = 20
fin = 100

ls_lv1 = lv1['ls']
ls0_lv2 = lv2['ls0']
ls1_lv2 = lv2['ls1']
ls2_lv2 = lv2['ls2']

var_lv1 = lv1['var']
var_lv2 = lv2['var']

x_prim_lv1 = lv1['x_prim']
x_prim_lv2 = lv2['x_prim']

Ai_lv1 = lv1['Ai']
Ai0_lv2 = lv2['Ai0']
Ai1_lv2 = lv2['Ai1']
Ai2_lv2 = lv2['Ai2']

Qi_lv1 = lv1['Qi']
Qi_lv2 = lv2['Qi']

n_basis_x = M1 ** nx
jv_x1 = np.zeros((n_basis_x, 1, nx))
lambda_x1 = np.zeros((n_basis_x, nx))
# trajectory from mean function
for ii in range(1, M1 + 1):
    for jj in range(1, M1 + 1):
        for kk in range(1, M1 + 1):
            ind = M1 ** 2 * (ii - 1) + M1 * (jj - 1) + kk
            jv_x1[ind - 1, 0, :] = np.array([ii, jj, kk])
            lambda_x1[ind - 1, :] = np.power(np.pi * jv_x1[ind - 1, :] / (2. * np.squeeze(L_x1)), 2)

phi_x1 = lambda x: np.prod(np.multiply(np.power(L_x1, -1 / 2), np.sin(np.pi * np.multiply(np.multiply(jv_x1, (np.reshape(x, (1, 1, nx)) + L_x1)), 1. / (2 * L_x1)))), axis=2)

jv_x2 = []
lambda_x2 = []
for ii in range(nx):
    jv, eigvals = se_kernel_eigvals(M2, L_x2[ii])
    jv_x2.append(jv)
    lambda_x2.append(eigvals)

phi_x2 = lambda x, idx: np.prod(np.multiply(np.power(L_x2[idx], -1 / 2), np.sin(np.pi * np.multiply(np.multiply(jv_x2[idx], (np.reshape(x, (1, 1, L_x2[idx].shape[2])) + L_x2[idx])), 1. / (2 * L_x2[idx])))), axis=2)

Ai_lv1_mean = np.mean(Ai_lv1[:, :, start:fin, :], axis=2)
Ai0_lv2_mean = np.mean(Ai0_lv2[:, :, start:fin, :], axis=2)
Ai1_lv2_mean = np.mean(Ai1_lv2[:, :, start:fin, :], axis=2)
Ai2_lv2_mean = np.mean(Ai2_lv2[:, :, start:fin, :], axis=2)
x_est_mean1 = np.zeros((nx, T, NEXP))
x_est_mean2 = np.zeros((nx, T, NEXP))

for iter in range(NEXP):
    x_est_mean1[:, 0, iter] = np.array([0.1, 0.8, 0.3])
    x_est_mean2[:, 0, iter] = np.array([0.1, 0.8, 0.3])
    for tt in range(1, T):
        x_est_mean1[:, tt, iter] = np.squeeze(Ai_lv1_mean[:, :, iter] @ phi_x1(X[tt - 1, :]))
        x_est_mean2[0, tt, iter] = Ai0_lv2_mean[:, :, iter] @ phi_x2(X[tt - 1, dims[0]], 0)
        x_est_mean2[1, tt, iter] = Ai1_lv2_mean[:, :, iter] @ phi_x2(X[tt - 1, dims[1]], 1)
        x_est_mean2[2, tt, iter] = Ai2_lv2_mean[:, :, iter] @ phi_x2(X[tt - 1, dims[2]], 2)
x_mc_mean1 = np.mean(x_est_mean1, axis=2)
x_mc_mean2 = np.mean(x_est_mean2, axis=2)

x_prim_lv1_mean = np.mean(x_prim_lv1[:, :, start:fin, :], axis=2)
x_prim_lv2_mean = np.mean(x_prim_lv2[:, :, start:fin, :], axis=2)
x_prim_lv1_mean_mc = np.mean(x_prim_lv1_mean, axis=2)
x_prim_lv2_mean_mc = np.mean(x_prim_lv2_mean, axis=2)

# trajectory error norm
rmse_lv1 = np.zeros((K + 1, nx, NEXP))
rmse_lv2 = np.zeros((K + 1, nx, NEXP))
for nn in range(NEXP):
    for kk in range(K + 1):
        for jj in range(nx):
            rmse_lv1[kk, jj, nn] = norm(X[:, jj] - x_prim_lv1[jj, :, kk, nn]) / T
            rmse_lv2[kk, jj, nn] = norm(X[:, jj] - x_prim_lv2[jj, :, kk, nn]) / T

rmse_lv1_mc = np.mean(rmse_lv1, axis=2)
rmse_lv2_mc = np.mean(rmse_lv2, axis=2)

# trajectory and errors
fig = plt.figure()
gs = GridSpec(3, 2, figure=fig)
ax00 = fig.add_subplot(gs[0, 0])
ax10 = fig.add_subplot(gs[1, 0])
ax20 = fig.add_subplot(gs[2, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax11 = fig.add_subplot(gs[1, 1])
ax21 = fig.add_subplot(gs[2, 1])

ax00.plot(t, X[:, 0], 'grey', lw=4., label='ground truth')
ax00.plot(t, x_mc_mean1[0, :], 'b--', label='SOTA', lw=2.)
ax00.plot(t, x_mc_mean2[0, :], 'r--', label='proposed', lw=2.)
ax00.set_ylabel('species 1[-]')
ax00.legend(prop={'size': 9})
ax00.grid()

ax10.plot(t, X[:, 1], 'grey', lw=4., label='ground truth')
ax10.plot(t, x_mc_mean1[1, :], 'b--', label='SOTA', lw=2.)
ax10.plot(t, x_mc_mean2[1, :], 'r--', label='proposed', lw=2.)
ax10.set_ylabel('species 2 [-]')
ax10.grid()

ax20.plot(t, X[:, 2], 'grey', lw=4., label='ground truth')
ax20.plot(t, x_mc_mean1[2, :], 'b--', label='SOTA', lw=2.)
ax20.plot(t, x_mc_mean2[2, :], 'r--', label='proposed', lw=2.)
ax20.set_ylabel('species 3 [-]')
ax20.set_xlabel('time [s]')
ax20.grid()

ax01.set_yscale('log')
ax01.plot(np.arange(fin + 1), rmse_lv1_mc[:fin + 1, 0], 'b.-', label='SOTA')
ax01.plot(np.arange(fin + 1), rmse_lv2_mc[:fin + 1, 0], 'r.-', label='proposed')
ax01.legend(prop={'size': 9})
ax01.set_ylabel('rmse [-]')
ax01.grid()

ax11.set_yscale('log')
ax11.plot(np.arange(fin + 1), rmse_lv1_mc[:fin + 1, 1], 'b.-', label='SOTA')
ax11.plot(np.arange(fin + 1), rmse_lv2_mc[:fin + 1, 1], 'r.-', label='proposed')
ax11.set_ylabel('rmse [-]')
ax11.grid()

ax21.set_yscale('log')
ax21.plot(np.arange(fin + 1), rmse_lv1_mc[:fin + 1, 2], 'b.-', label='SOTA')
ax21.plot(np.arange(fin + 1), rmse_lv2_mc[:fin + 1, 2], 'r.-', label='proposed')
ax21.set_ylabel('rmse [-]')
ax21.set_xlabel('Iterations [-]')
ax21.grid()

# fig.suptitle('Lotka-Volterra Model')
plt.tight_layout()

plt.show()


# mean trajectory
fig0, axs0 = plt.subplots(3, 1)
axs0[0].plot(t, X[:, 0], 'grey', lw=4., label='ground truth')
axs0[0].plot(t, x_mc_mean1[0, :], 'b--', label='SOTA mean')
axs0[0].plot(t, x_mc_mean2[0, :], 'r--', label='proposed mean')
axs0[0].plot(t, x_prim_lv1_mean_mc[0, :], 'k--', label='SOTA traj. mean')
axs0[0].plot(t, x_prim_lv2_mean_mc[0, :], 'g--', label='proposed traj.mean')
axs0[0].set_ylabel('species 1[-]')
axs0[0].legend(prop={'size': 9})
axs0[0].grid()

axs0[1].plot(t, X[:, 1], 'grey', lw=4., label='ground truth')
axs0[1].plot(t, x_mc_mean1[1, :], 'b--', label='SOTA mean')
axs0[1].plot(t, x_mc_mean2[1, :], 'r--', label='proposed mean')
axs0[1].plot(t, x_prim_lv1_mean_mc[1, :], 'k--', label='SOTA traj. mean')
axs0[1].plot(t, x_prim_lv2_mean_mc[1, :], 'g--', label='proposed traj.mean')
axs0[1].set_ylabel('species 2 [-]')
axs0[1].grid()

axs0[2].plot(t, X[:, 2], 'grey', lw=4., label='ground truth')
axs0[2].plot(t, x_mc_mean1[2, :], 'b--', lw=2., label='SOTA mean')
axs0[2].plot(t, x_mc_mean2[2, :], 'r--', lw=2., label='proposed mean')
axs0[2].plot(t, x_prim_lv1_mean_mc[2, :], 'k--', label='SOTA traj. mean')
axs0[2].plot(t, x_prim_lv2_mean_mc[2, :], 'g--', label='proposed traj.mean')
axs0[2].set_ylabel('species 3 [-]')
axs0[2].set_xlabel('time [s]')
axs0[2].grid()

print('finished!')