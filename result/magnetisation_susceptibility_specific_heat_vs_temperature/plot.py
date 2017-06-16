import numpy as np
import matplotlib.pyplot as plt
import fileinput


# read data
T = []
Mx = []
My = []
M = []
Chi = []
E = []
Cv = []
for line in fileinput.input():
    if fileinput.filelineno() == 1:
        L = int(line.rstrip('\n'))
    elif fileinput.filelineno() == 2:
        n_sample = int(line.rstrip('\n'))
    elif fileinput.filelineno() == 3:
        n_warm_mutiple = int(line.rstrip('\n'))
    elif fileinput.filelineno() == 4:
        T.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 5:
        Mx.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 6:
        My.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 7:
        M.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 8:
        Chi.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 9:
        E.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 10:
        Cv.append(np.array(line.rstrip(',\n').split(',')).astype(float))

T = np.reshape(np.array(T),np.size(T))
Mx = np.reshape(np.array(Mx),np.size(Mx))
My = np.reshape(np.array(My),np.size(My))
M = np.reshape(np.array(M),np.size(M))
Chi = np.reshape(np.array(Chi),np.size(Chi))
E = np.reshape(np.array(E),np.size(E))
Cv = np.reshape(np.array(Cv),np.size(Cv))

save_fig_name = "fig_" + str(L) + "_" + str(n_sample) + "_" + str(n_warm_mutiple) + ".eps"
n_warm_log2 = np.log2(n_warm_mutiple*L*L)
fig_title = r'size: $' + str(L) + r'\times' + str(L) + '$' + r'   $n_{sample} = ' + str(n_sample) + '$' + \
    '    warm up steps: $2^{' + str(n_warm_log2) + '} = ' + str(L) + r'\times' + str(L) + r'\times' + str(n_warm_mutiple) + '=' + str(L*L*n_warm_mutiple) + '$'

'''
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
'''

plt.figure(figsize=(10, 30))

plt.subplot(5,1,1)
plt.title(fig_title)
line1, = plt.plot(T, Mx, label= r'$M_x$')
line2, = plt.plot(T, My, label= r'$M_y$')
plt.legend(handles=[line1, line2])
plt.grid()
plt.ylabel(r'$\vec{M}$')

plt.subplot(5,1,2)
plt.plot(T, Chi, color='b', marker='o', ms=2)
plt.grid()
plt.ylabel(r'$|\vec{M}|$')

plt.subplot(5,1,3)
plt.plot(T, Chi, color='g', marker='o', ms=2)
plt.grid()
plt.ylabel(r'$\chi$')

plt.subplot(5,1,4)
plt.plot(T, E, color = 'm', marker='o', ms=2)
plt.grid()
plt.ylabel(r'$E$')

plt.subplot(5,1,5)
plt.plot(T, Cv, color = 'r', marker='o', ms=2)
plt.grid()
plt.ylabel(r'$C_v$')
plt.xlabel(r'$T$')

plt.tight_layout()
plt.subplots_adjust(hspace=0)


plt.savefig(save_fig_name)
