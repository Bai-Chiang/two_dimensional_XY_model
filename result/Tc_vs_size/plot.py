import numpy as np
import matplotlib.pyplot as plt
import fileinput


# read
L = []
T = []
Tc = []
Cv = []
n_sample = []
for line in fileinput.input():
    if fileinput.filelineno() == 1:
        n_warm_mutiple = int(line.rstrip('\n'))
    elif fileinput.filelineno() == 2:
        L.append(np.array(line.rstrip(',\n').split(',')).astype(int))
    elif fileinput.filelineno() == 3:
        n_sample.append(np.array(line.rstrip(',\n').split(',')).astype(int))
    elif fileinput.filelineno() == 4:
        Tc.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    elif fileinput.filelineno() == 5:
        T.append(np.array(line.rstrip(',\n').split(',')).astype(float))
    else:
        Cv.append(np.array(line.rstrip(',\n').split(',')).astype(float))

n_sample = np.reshape(np.array(n_sample),np.size(n_sample))
T = np.reshape(np.array(T),np.size(T))
Tc = np.reshape(np.array(Tc),np.size(Tc))
L = np.reshape(np.array(L),np.size(L))
Cv = np.array(Cv)

save_fig_name = "fig_" + str(n_sample[np.size(n_sample)-1]) + "_" + str(n_warm_mutiple) + ".eps"
#n_warm_log2 = np.log2(n_warm_mutiple*L*L)
fig_title =  r'warm up steps: length $\times$ length $\times$' + str(n_warm_mutiple)



plt.figure(figsize=(8, 10))

plt.subplot(2,1,1)
plt.title(fig_title)
line0, = plt.plot(T, Cv[0], color='b', marker='o', ms=1, lw=1, label= 'length='+str(L[0])+r'    $n_{sample}=$'+str(n_sample[0]))
line1, = plt.plot(T, Cv[1], color='c', marker='o', ms=1, lw=1, label= 'length='+str(L[1])+r'    $n_{sample}=$'+str(n_sample[1]))
line2, = plt.plot(T, Cv[2], color='y', marker='o', ms=1, lw=1, label= 'length='+str(L[2])+r'    $n_{sample}=$'+str(n_sample[2]))
line3, = plt.plot(T, Cv[3], color='g', marker='o', ms=1, lw=1, label= 'length='+str(L[3])+r'    $n_{sample}=$'+str(n_sample[3]))
line4, = plt.plot(T, Cv[4], color='m', marker='o', ms=1, lw=1, label= 'length='+str(L[4])+r'    $n_{sample}=$'+str(n_sample[4]))
line5, = plt.plot(T, Cv[5], color='r', marker='o', ms=1, lw=1, label= 'length='+str(L[5])+r'    $n_{sample}=$'+str(n_sample[5]))
plt.legend(handles=[line0, line1, line2, line3, line4, line5])
plt.grid()
plt.ylabel(r'$C_v$')
plt.xlabel(r'$T$')


plt.subplot(2,1,2)
plt.scatter(L, Tc)
plt.grid()
plt.ylabel(r'$T_c$')
plt.xlabel('length')
plt.xticks(L)

plt.tight_layout()


plt.savefig(save_fig_name)
