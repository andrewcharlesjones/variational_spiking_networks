import numpy as np
from scipy.stats import bernoulli


def staircase(num_neurons, num_timebins, stair_range = [10, 30], random_bit_flip_rate = 0.05):
	'''
	staircase pattern
	'''
	neuron_block_size = np.floor(num_neurons / 3)
	data = np.zeros((num_neurons, num_timebins))
	curr_neuron_block = 0

	curr_stair_idx = 0
	while curr_stair_idx <= num_timebins:
		curr_stair_length = np.int(np.random.uniform(low=stair_range[0], high = stair_range[1]))
		data[int(curr_neuron_block * neuron_block_size):int((curr_neuron_block + 1)* neuron_block_size), curr_stair_idx:curr_stair_idx + curr_stair_length] = 1
		if curr_neuron_block < 2:
			curr_neuron_block += 1
		else:
			curr_neuron_block = 0
		curr_stair_idx += curr_stair_length

	# randomly flip some bits
	bit_flip_mask = np.random.binomial(n = 1, p = random_bit_flip_rate, size=(num_neurons, num_timebins))
	data[bit_flip_mask.astype(bool)] = np.abs(data[bit_flip_mask.astype(bool)] - 1)
	
	return data
