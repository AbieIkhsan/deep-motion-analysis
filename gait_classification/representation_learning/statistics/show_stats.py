import matplotlib.pyplot as plt
import numpy as np

vae_gan_repr_cost_mean = np.load('vae_gan_stats.npz')['repr_cost_mean']
aae_repr_cost_mean = np.load('ae_gan_stats.npz')['repr_cost_mean']

v_repr_plot, = plt.plot(vae_gan_repr_cost_mean, 'r', label='VAE-GAN')
a_repr_plot, = plt.plot(aae_repr_cost_mean, 'b', label='AAE')

plt.legend(handles=[v_repr_plot, a_repr_plot,])
plt.show()