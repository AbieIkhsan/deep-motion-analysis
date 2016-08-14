import matplotlib.pyplot as plt
import numpy as np

enc_cost_mean = np.load('vae_gan_stats.npz')['enc_cost_mean']
x_axis = np.arange(10)
plt.plot(x_axis, enc_cost_mean, 'r^')
plt.show()