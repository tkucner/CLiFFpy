import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import helpers as he
from cl_map import CLCellType
from cl_map import CLMap as cl

# line format
# time [ms] (unixtime + milliseconds/1000), person id, position x [mm], position y [mm], position z (height) [mm], velocity [mm/s], angle of motion [rad], facing angle [rad]
# time, person_id, x, y, z, velocity, motion_angle, facing_angle

file_name = "/home/tzkr/Data/Directional/ATC/atc-20121114.csv"

chunksize = 10 ** 2
loop_number = 0
refresh_ratio = 1
x_min = -45000
x_max = 45000
y_min = -30000
y_max = 30000
step = 1000
p_shape = (round((y_max - y_min) / step), round((x_max - x_min) / step))

cl_map = cl()

cl_map.set_up_map(step=step, processing=CLCellType.STREAM)

fig0, ax0 = plt.subplots(1, 1)  # ,sharex=True,sharey=True)
# fig1, ax1= plt.subplots(1, 1)#,sharex=True,sharey=True)

for chunk in pd.read_csv(file_name, chunksize=chunksize):
    cl_map.load_data(chunk)
    loop_number = loop_number + 1
    plot_data = []
    if loop_number % refresh_ratio == 0:
        ax0.clear()
        ax0.set_aspect('equal')
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_min, y_max)
        cl_map.cluster_data()

        p_array_vis = np.zeros(p_shape)
        # visualisation
        for cell in cl_map.cells_data:
            for m in cell.clustering_results.mean_values:
                u, v = he.pol2cart(m[0], m[1])
                row = [cell.corner[0] + cl_map.grid_step / 2, cell.corner[1] + cl_map.grid_step / 2, u, v]
                plot_data.append(row)
                p_array_vis[int((cell.corner[1] - y_min) / step), int((cell.corner[0] - x_min) / step)] = cell.count[
                                                                                                              0] / cl_map.total_number_of_observations
        plot_data = np.array(plot_data)
        ax0.quiver(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], plot_data[:, 3], units='xy')
        ax0.set_title("Observations count: " + str(chunksize * loop_number))
        plt.savefig("directions_" + str(loop_number).zfill(5) + '.png', bbox_inches='tight')

        ax0.clear()
        ax0.set_aspect('equal')
        ax0.set_title("Observations count: " + str(chunksize * loop_number))
        ax0.imshow(np.flipud(p_array_vis), cmap="plasma", vmin=0,vmax=np.max(np.max(p_array_vis)))
        plt.savefig("intensity_" + str(loop_number).zfill(5) + '.png', bbox_inches='tight')
