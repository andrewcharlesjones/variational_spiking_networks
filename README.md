# variational_spiking_networks

Python implementation and sample tests of performing variational inference in recurrent spiking networks.

Reference: Stochastic variational learning in recurrent spiking networks by Rezende and Gerstner (2014) https://doi.org/10.3389/fncom.2014.00038

Description of files:
- glm_spiking_visible_only.ipynb: Fits a GLM with only visible neurons.
- mse_test_increasing_data.py: Computes MSE between true and learned parameters in the viisble-only case, and shows how this error declines as the network sees more data.
- vi_spiking_visible_and_hidden.ipynb: Fits a GLM with visible and hidden neurons using variational inference.
- rezende_gradient_updates.pdf: PDF of gradient updates with more detail than provided in the paper.
- stimuli.py: Ways to generate different data types (still in dev)
