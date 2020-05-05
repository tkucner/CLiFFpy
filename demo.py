import cl_rand
import cl_visualisation as clvis
import numpy as np
import mean_shift as ms
#import cl_arithmetic as cla
import matplotlib.pyplot as plt

#r = cl_rand.cl_gauss_2d([6.28, 1.2], [[0.1, 0.0], [0.0, 0.1]], 1000)
#r = np.append(r, cl_rand.cl_gauss_2d([3.14, 1.2], [[0.1, 0.0], [0.0, 0.1]], 1000), axis=0)
r = cl_rand.cl_gauss_2d([3.14, 1.2], [[0.1, 0.0], [0.0, 0.1]], 1000)

mean_shifter = ms.MeanShift()
mean_shift_result = mean_shifter.cluster(r, kernel_bandwidth=0.5)

original_points = mean_shift_result.original_points
shifted_points = mean_shift_result.shifted_points
cluster_assignments = mean_shift_result.cluster_ids
history = mean_shift_result.history

x = original_points[:, 0]
y = original_points[:, 1]
Cluster = cluster_assignments
centers = shifted_points

fig = plt.figure()
ax = fig.add_subplot(211)

for it in range(0,len(history)):

    l_hist = history[it]
    l_hist = np.array(l_hist)
    plt.text(l_hist[0, 0], l_hist[0, 1], str(it))
    X = l_hist[:, 0]
    Y = l_hist[:, 1]
    ax.plot(X, Y)
    ax.scatter(X, Y, s=5)

bx = fig.add_subplot(212)
scatter = bx.scatter(x, y, c=Cluster, s=20)
for i,j in centers:
    bx.scatter(i, j, s=500, c='red', marker='+')


plt.show()

