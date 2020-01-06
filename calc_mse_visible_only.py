import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import pandas as pd
from scipy import optimize

tau = 10.
eta_0 = 0.1
tau_adapt = 10.
rho_0 = 0.5
v = 0
delta_u = 1.
mu_M = 0.00001
mu_Q = 0.00001
tau_G = 10.
tau_baseline = 100.
dt = 1.

num_neurons = 30
# num_timebins = 1000

num_timebins_list = [10, 100, 1000, 10000]

mse_list = []

for num_timebins in num_timebins_list:
    curr_mse_list = []

    print "\n\n=== {} timebins ===".format(num_timebins)

    for _ in range(5):

        weights = np.random.normal(loc = 0, scale = 0.1, size = (num_neurons, num_neurons))

        np.fill_diagonal(weights, 0)
        phi = np.zeros(shape = (num_neurons, num_timebins))
        rho = np.zeros(shape = (num_neurons, num_timebins))
        eta = np.zeros(shape = (num_neurons, num_timebins))
        u = np.zeros(shape = (num_neurons, num_timebins))
        spikes = np.zeros(shape = (num_neurons, num_timebins))
        spike_probs = np.zeros(shape = (num_neurons, num_timebins))

        phi[:, 0] = np.zeros(num_neurons)
        eta[:, 0] = np.zeros(num_neurons)
        u[:, 0] = np.matmul(weights, phi[:, 0]) - eta[:, 0]
        rho[:, 0] = rho_0 * np.exp(u[:, 0])
        spike_probs[:, 0] = 1-np.exp(-rho[:, 0])
        spikes[:, 0] = np.random.binomial(n = 1, p = spike_probs[:, 0])






        for tt in range(1, num_timebins):
                
            for ii in range(num_neurons):
                
                dphi = dt * (1 / tau) * (spikes[:, tt - 1] - phi[:, tt - 1])
                deta = dt * (1 / tau_adapt) * (spikes[:, tt - 1] - eta[:, tt - 1])
                
                phi[:, tt] = phi[:, tt - 1] + dphi
                eta[:, tt] = eta[:, tt - 1] + deta
                        
                total_potential = np.matmul(weights, phi[:, tt]) - eta_0 * eta[:, tt]
                u[:, tt] = total_potential
                
                rho[:, tt] = rho_0 * np.exp((u[:, tt] - v) / delta_u)
                
                spike_probs[:, tt] = 1-np.exp(-dt*rho[:, tt])
                spikes[:, tt] = np.random.binomial(n = 1, p = spike_probs[:, tt])


        def log_likelihood(curr_weights):

            # total_potential = np.matmul(curr_weights, phi) - eta_0 * eta[0, :]

            # rho[0, :] = rho_0 * np.exp((total_potential - v) / delta_u)
            # LL = np.sum(np.multiply(np.log(rho[0, :]), spikes[0, :]) - rho[0, :]) * dt

            curr_weights = np.reshape(curr_weights, [num_neurons, num_neurons])

            total_potential = np.matmul(curr_weights, phi) - eta_0 * eta

            rho = rho_0 * np.exp((total_potential - v) / delta_u)
            LL = np.sum(np.multiply(np.log(rho), spikes) - rho) * dt
            return -LL



        # x0 = np.random.normal(loc=0, scale=0.1, size=num_neurons)
        x0 = np.random.normal(loc=0, scale=0.1, size=[num_neurons, num_neurons])
        res = optimize.minimize(log_likelihood, x0, options={'disp': False}, tol=1e-5)

        # curr_mse = np.sum((res.x - weights[0, :])**2)
        curr_mse = np.sum((np.reshape(res.x, [num_neurons, num_neurons]) - weights)**2)
        print curr_mse
        curr_mse_list.append(curr_mse)

    mse_list.append(curr_mse_list)


mse_list = np.array(mse_list)
timebins = np.arange(len(num_timebins_list))

plt.bar(timebins, np.mean(mse_list, axis = 1), yerr=np.std(mse_list, axis=1))
plt.yscale('log')
plt.xlabel("Num timebins")
plt.ylabel("MSE, true/fitted weights")
plt.xticks(np.arange(len(num_timebins_list)), [10**x for x in range(1, len(num_timebins_list) + 1)], rotation=0)
plt.savefig("./plots/mse_visible_only.png")
plt.show()
import ipdb; ipdb.set_trace()
