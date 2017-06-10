import numpy as np
import matplotlib.pyplot as plt

save_fig_name = "quiver_plot_T_0.5.eps"
fig_title = r"size: $128 \times 128$   warm up steps: $2^{19} \approx 5 \times 10^{5}$"

# read data
file = open('quiver_plot_T_0.5.csv', 'r')
theta = []
for line in file:
    theta.append(np.array(line.rstrip(',\n').split(',')).astype(float))
theta = np.array(theta)


# calculate x components and y components for all spins
Sx = np.cos(theta)
Sy = np.sin(theta)

# plot figure
Y, X = np.mgrid[0:np.shape(theta)[0], 0:np.shape(theta)[1]]
plt.figure(figsize=(8, 8))
plt.rcParams['image.cmap'] = 'Dark2'
plt.quiver(X, Y, Sx, Sy, theta, pivot='mid')
plt.axis('off')
plt.title(fig_title, fontsize = 10)

plt.savefig(save_fig_name)
