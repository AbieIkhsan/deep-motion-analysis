import matplotlib.pyplot as plt
import numpy as np

repr_cost_mean = np.load('vae_gan_stats_cmu.npz')['repr_cost_mean']
vari_cost_mean = np.load('vae_gan_stats_cmu.npz')['vari_cost_mean']
disc_cost_mean = np.load('vae_gan_stats_cmu.npz')['disc_cost_mean']

repr_plot, = plt.plot(repr_cost_mean, 'r', label='Repr. Error')
#vari_plot, = plt.plot(vari_cost_mean, 'b', label='Vari. Error')
disc_plot, = plt.plot(disc_cost_mean, 'g', label='Disc. Error')


plt.legend(handles=[repr_plot, disc_plot])
plt.show()