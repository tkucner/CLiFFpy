import matplotlib.pyplot as plt
import numpy as np

class CLVisualisation:
    def __init__(self,cl_map):
        self.map=cl_map

    def show_raw_locations(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(self.map.data['x'],self.map.data['y'])
        plt.show()

    def show_discretised_locations(self):
        cells=self.map.data['corners'].unique()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for cell in self.map.cells_data:
            ax.scatter(cell.data['x'],cell.data['y'])
        ax.set_xticks(
            np.arange(self.map.data_extent['x_min'], self.map.data_extent['x_max'] + 2 * self.map.grid_step,
                      self.map.grid_step))
        ax.set_yticks(
            np.arange(self.map.data_extent['y_min'], self.map.data_extent['y_max'] + 2 * self.map.grid_step,
                      self.map.grid_step))


        plt.grid()
        plt.plot()
