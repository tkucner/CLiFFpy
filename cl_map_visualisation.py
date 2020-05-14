import matplotlib.pyplot as plt
import numpy as np

import helpers as he


class CLVisualisation:
    def __init__(self, cl_map):
        self.map = cl_map

    def show_raw_locations(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(self.map.data['x'], self.map.data['y'])
        plt.show()

    def show_discretised_locations(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for cell in self.map.cells_data:
            ax.scatter(cell.data['x'], cell.data['y'])
        ax.set_xticks(
            np.arange(self.map.data_extent['x_min'], self.map.data_extent['x_max'] + 2 * self.map.grid_step,
                      self.map.grid_step))
        ax.set_yticks(
            np.arange(self.map.data_extent['y_min'], self.map.data_extent['y_max'] + 2 * self.map.grid_step,
                      self.map.grid_step))
        plt.grid()
        plt.plot()

    def show_directions(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plot_data = []
        for cell in self.map.cells_data:
            for m in cell.clustering_results.mean_values:
                u, v = he.pol2cart(m[0], m[1])
                row = [cell.corner[0] + self.map.grid_step / 2, cell.corner[1] + self.map.grid_step / 2, u, v]
                plot_data.append(row)
        plot_data = np.array(plot_data)
        ax.quiver(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], plot_data[:, 3],units='xy')
        plt.show()
