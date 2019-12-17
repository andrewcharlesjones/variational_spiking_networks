import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


num_neurons = 2
num_timesteps = 200

tau = 0.01
tau_adapt = 0.01
eta_0 = 0.1
rho_0 = 1
v = 0
delta_u = 1
dt = 0.001

# w = np.array([[1, 1], [1, 0.5]])
w = np.random.normal(loc = 0, scale = 0.1, size = [num_neurons, num_neurons])

spikes = np.zeros((num_neurons, num_timesteps))
potentials = np.zeros((num_neurons, num_timesteps))

initial_potential = [0.1, 0.1]
potentials[:, 0] = initial_potential

spikes[:, 0] = [1, 0]


for tt in range(1, num_timesteps):

	### TIME IN MILLISECONDS
	tt_ms = tt / 1000.

	### FILTER INCOMING SPIKES
	time_array = np.arange(tt) / 1000.
	time_filter = np.exp(-(tt_ms - time_array) / tau)
	filtered_spikes = np.matmul(spikes[:, :tt], time_filter) * dt # phi_j

	### WEIGHT INCOMING SPIKES BY SYNAPTIC WEIGHT
	weighted_spikes = np.matmul(w, filtered_spikes)

	### APPLY POST-SPIKE ADAPTATION
	adaptation_filter = np.exp(-(tt_ms - time_array) / tau_adapt)
	filtered_adaptation = np.matmul(spikes[:, :tt], adaptation_filter) * dt * -eta_0

	### CURRENT MEMBRANE POTENTIAL
	curr_potentials = weighted_spikes + filtered_adaptation

	### FIRING
	firing_intensity = rho_0 * np.exp((curr_potentials - v) / delta_u)
	curr_spikes = np.random.poisson(firing_intensity)

	print curr_potentials
	print curr_spikes

	potentials[:, tt] = curr_potentials
	spikes[:, tt] = curr_spikes

	# import ipdb; ipdb.set_trace()

plt.plot(potentials[0, 1:])
plt.plot(potentials[1, 1:])
plt.show()

fig, ax = plt.subplots(figsize=(15,1))  
sns.heatmap(spikes)
plt.show()
import ipdb; ipdb.set_trace()
