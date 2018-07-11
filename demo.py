import cl_rand
import cl_visualisation as cl_vis
import numpy as np
import mean_shift as ms
import cl_aritmetics as cl_a
import matplotlib.pyplot as plt

#r = cl_rand.cl_gauss_2d([1.5, 1.2], [[0.1, 0.02], [0.1, 0.1]], 1000)
#r = np.append(r, cl_rand.cl_gauss_2d([6, 2.2], [[0.1, -0.02], [0.1, 0.1]], 1000), axis=0)
r = cl_rand.cl_gauss_2d([6, 2.2], [[0.1, -0.02], [0.1, 0.1]], 1000)


cl_vis.plot_data_simple(r)



mean_shifter = ms.MeanShift()
mean_shift_result = mean_shifter.cluster(r, kernel_bandwidth = 1)
print(mean_shift_result)

original_points =  mean_shift_result.original_points
shifted_points = mean_shift_result.shifted_points
cluster_assignments = mean_shift_result.cluster_ids

x = original_points[:,0]
y = original_points[:,1]
Cluster = cluster_assignments
centers = shifted_points

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,c=Cluster,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.savefig("mean_shift_result")

