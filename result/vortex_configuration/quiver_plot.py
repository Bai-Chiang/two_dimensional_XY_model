import numpy as np
import matplotlib.pyplot as plt
import fileinput


# read data
theta = []
for line in fileinput.input():
    if fileinput.filelineno() == 1:
        L = int(line.rstrip('\n'))
    elif fileinput.filelineno() == 2:
        T = float(line.rstrip('\n'))
    elif fileinput.filelineno() == 3:
        n_warm = int(line.rstrip('\n'))
    else:
        theta.append(np.array(line.rstrip(',\n').split(',')).astype(float))
theta = np.array(theta)


save_fig_name = "fig_" + str(L) + "_" + str(T) + "_" + str(n_warm) + ".eps"
n_warm_log2 = np.log2(n_warm)
fig_title = r'size: $' + str(L) + r'\times' + str(L) + '$    $T = ' + str(T) + '$' + '    warm up steps: $2^{' + str(n_warm_log2) + '} = ' + str(n_warm) + '$'


# calculate x components and y components for all spins
Sx = np.cos(theta)
Sy = np.sin(theta)

# plot figure
Y, X = np.mgrid[0:np.shape(theta)[0], 0:np.shape(theta)[1]]
plt.figure(figsize=(8, 8))
plt.rcParams['image.cmap'] = 'hsv'
plt.quiver(X, Y, Sx, Sy, theta, pivot='mid')
plt.axis('off')
plt.title(fig_title, fontsize = 10)

plt.savefig(save_fig_name)

