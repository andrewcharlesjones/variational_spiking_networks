import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


num_neurons = 2
num_timesteps = 2000

tau = 0.01
tau_adapt = 0.01
eta_0 = 0.1
rho_0 = 1
v = 0
delta_u = 1
dt = 0.001
mu_M = 1000

# w = np.array([[1, 1], [1, 0.5]])
w = np.random.normal(loc = 0, scale = 0.1, size = [num_neurons, num_neurons])

# train data
spikes = np.zeros((num_neurons, num_timesteps))
potentials = np.zeros((num_neurons, num_timesteps))
true_firing_intensities = np.zeros((num_neurons, num_timesteps))
initial_potential = np.random.normal(loc = 0, scale = 0.1, size = num_neurons)
potentials[:, 0] = initial_potential
firing_intensity = rho_0 * np.exp((potentials[:, 0] - v) / delta_u) # rho_i
true_firing_intensities[:, 0] = firing_intensity
spikes[:, 0] = np.random.poisson(lam = firing_intensity, size = num_neurons)

# test data
test_spikes = np.zeros((num_neurons, num_timesteps))
test_potentials = np.zeros((num_neurons, num_timesteps))
test_true_firing_intensities = np.zeros((num_neurons, num_timesteps))
test_initial_potential = np.random.normal(loc = 0, scale = 0.1, size = num_neurons)
test_potentials[:, 0] = test_initial_potential
test_firing_intensity = rho_0 * np.exp((test_potentials[:, 0] - v) / delta_u) # rho_i
test_true_firing_intensities[:, 0] = test_firing_intensity
test_spikes[:, 0] = np.random.poisson(lam = test_firing_intensity, size = num_neurons)

for tt in range(1, num_timesteps):

	### TIME IN MILLISECONDS
	tt_ms = tt / 1000.

	### FILTER INCOMING SPIKES
	time_array = np.arange(tt) / 1000.
	time_filter = np.exp(-(tt_ms - time_array) / tau)
	filtered_spikes = np.matmul(spikes[:, :tt], time_filter) * dt # phi_j
	test_filtered_spikes = np.matmul(test_spikes[:, :tt], time_filter) * dt # phi_j

	### WEIGHT INCOMING SPIKES BY SYNAPTIC WEIGHT
	weighted_spikes = np.matmul(w, filtered_spikes)
	test_weighted_spikes = np.matmul(w, test_filtered_spikes)

	### APPLY POST-SPIKE ADAPTATION
	adaptation_filter = np.exp(-(tt_ms - time_array) / tau_adapt)
	filtered_adaptation = np.matmul(spikes[:, :tt], adaptation_filter) * dt * -eta_0 # eta_i
	test_filtered_adaptation = np.matmul(test_spikes[:, :tt], adaptation_filter) * dt * -eta_0 # eta_i

	### CURRENT MEMBRANE POTENTIAL
	curr_potentials = weighted_spikes + filtered_adaptation # u_i
	test_curr_potentials = test_weighted_spikes + test_filtered_adaptation # u_i

	### FIRING
	firing_intensity = rho_0 * np.exp((curr_potentials - v) / delta_u) # rho_i
	true_firing_intensities[:, tt] = firing_intensity
	curr_spikes = np.random.poisson(firing_intensity)

	test_firing_intensity = rho_0 * np.exp((test_curr_potentials - v) / delta_u) # rho_i
	test_true_firing_intensities[:, tt] = test_firing_intensity
	test_curr_spikes = np.random.poisson(test_firing_intensity)

	potentials[:, tt] = curr_potentials
	spikes[:, tt] = curr_spikes

	test_potentials[:, tt] = test_curr_potentials
	test_spikes[:, tt] = test_curr_spikes


fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(potentials[0, 1:])
plt.plot(potentials[1, 1:])
plt.savefig("./plots/true_weights_potentials.png")
# plt.show()

fig, ax = plt.subplots(figsize=(10, 3))  
sns.heatmap(spikes, xticklabels=False, yticklabels=False)
plt.title("Observed spikes (simulated using random weights)")
plt.ylabel("Neuron")
plt.savefig("./plots/true_weights_spikes.png")
# plt.show()
import ipdb; ipdb.set_trace()




num_train_iters = 10000
learned_w = np.random.normal(loc = 0, scale = 0.1, size = [num_neurons, num_neurons])

learned_potentials = np.zeros((num_neurons, num_timesteps))
initial_potential = np.random.normal(loc = 0, scale = 0.1, size = num_neurons)
learned_potentials[:, 0] = initial_potential


LL_trace = []
for iter_num in range(num_train_iters):

	### UPDATE POTENTIALS
	for tt in range(1, num_timesteps):

		### TIME IN MILLISECONDS
		tt_ms = tt / 1000.

		### FILTER INCOMING SPIKES
		time_array = np.arange(tt) / 1000.
		time_filter = np.exp(-(tt_ms - time_array) / tau)
		filtered_spikes = np.matmul(spikes[:, :tt], time_filter) * dt # phi_j

		### WEIGHT INCOMING SPIKES BY SYNAPTIC WEIGHT
		weighted_spikes = np.matmul(learned_w, filtered_spikes)

		### APPLY POST-SPIKE ADAPTATION
		adaptation_filter = np.exp(-(tt_ms - time_array) / tau_adapt)
		filtered_adaptation = np.matmul(spikes[:, :tt], adaptation_filter) * dt * -eta_0 # eta_i

		### CURRENT MEMBRANE POTENTIAL
		curr_potentials = weighted_spikes + filtered_adaptation # u_i

		### FIRING
		firing_intensity = rho_0 * np.exp((curr_potentials - v) / delta_u) # rho_i
		curr_spikes = np.random.poisson(firing_intensity)

		learned_potentials[:, tt] = curr_potentials

	### UPDATE WEIGHTS
	mean_spike_diffs = spikes - rho_0 * np.exp((learned_potentials - v) / delta_u)
	filtered_spikes_arr = []

	for tt in range(num_timesteps):

		### TIME IN MILLISECONDS
		tt_ms = tt / 1000.

		time_array = np.arange(tt) / 1000.

		filtered_spikes = np.matmul(spikes[:, :tt], np.exp(-(tt_ms - time_array) / tau)) * dt
		filtered_spikes_arr.append(filtered_spikes)

	weight_gradient = np.matmul(mean_spike_diffs, np.array(filtered_spikes_arr)) * dt
	learned_w += mu_M * weight_gradient

	# print learned_w
	# import ipdb; ipdb.set_trace()


	### LOG LIKELIHOOD

	firing_intensities = rho_0 * np.exp((learned_potentials - v) / delta_u)
	individual_LL = (np.multiply(np.log(firing_intensities), spikes) - firing_intensities) * dt
	LL = np.sum(individual_LL)
	LL_trace.append(LL)
	if iter_num % 10 == 0:
		print 'Train LL: {}'.format(LL)

	firing_intensities = rho_0 * np.exp((learned_potentials - v) / delta_u)
	individual_LL = (np.multiply(np.log(firing_intensities), test_spikes) - firing_intensities) * dt
	LL = np.sum(individual_LL)
	if iter_num % 10 == 0:
		print 'Test LL: {}'.format(LL)


fig, ax = plt.subplots(figsize=(7, 7))
plt.plot(LL_trace)
plt.xlabel("timestep")
plt.ylabel("log-likelihood")
plt.savefig("./plots/LL_trace.png")


### SIMULATE SPIKES FROM TRAINED MODEL

spikes = np.zeros((num_neurons, num_timesteps))
potentials = np.zeros((num_neurons, num_timesteps))

initial_potential = np.repeat(0.1, num_neurons)
potentials[:, 0] = initial_potential

spikes[:, 0] = np.random.binomial(n = 1, p = 0.5, size = num_neurons)


for tt in range(1, num_timesteps):

	### TIME IN MILLISECONDS
	tt_ms = tt / 1000.

	### FILTER INCOMING SPIKES
	time_array = np.arange(tt) / 1000.
	time_filter = np.exp(-(tt_ms - time_array) / tau)
	filtered_spikes = np.matmul(spikes[:, :tt], time_filter) * dt # phi_j

	### WEIGHT INCOMING SPIKES BY SYNAPTIC WEIGHT
	weighted_spikes = np.matmul(learned_w, filtered_spikes)

	### APPLY POST-SPIKE ADAPTATION
	adaptation_filter = np.exp(-(tt_ms - time_array) / tau_adapt)
	filtered_adaptation = np.matmul(spikes[:, :tt], adaptation_filter) * dt * -eta_0 # eta_i

	### CURRENT MEMBRANE POTENTIAL
	curr_potentials = weighted_spikes + filtered_adaptation # u_i

	### FIRING
	firing_intensity = rho_0 * np.exp((curr_potentials - v) / delta_u) # rho_i
	curr_spikes = np.random.poisson(firing_intensity)

	# print curr_potentials
	# print curr_spikes

	potentials[:, tt] = curr_potentials
	spikes[:, tt] = curr_spikes


fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(potentials[0, 1:])
plt.plot(potentials[1, 1:])
plt.savefig("./plots/learned_weights_potentials.png")
# plt.show()

fig, ax = plt.subplots(figsize=(10,3))
sns.heatmap(spikes, xticklabels=False, yticklabels=False)
plt.title("Simulated spikes (using fitted weights)")
plt.ylabel("Neuron")
plt.savefig("./plots/learned_weights_spikes.png")
# plt.show()
import ipdb; ipdb.set_trace()

		



