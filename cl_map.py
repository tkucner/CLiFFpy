import multiprocessing as mp
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

import mean_shift as ms


class CLCellShape(Enum):
    CIRCULAR = 1
    SQUARE = 2


class CLCellType(Enum):
    BATCH = 1
    STREAM = 2


class ClusteringType(Enum):
    MS = 1


def cluster_worker(obj):
    obj.query()
    return obj


class CLCell:
    def __init__(self, corner, data, clustering_type, cell_shpae, kernel_bandwidth=0.5, cell_type=CLCellType.BATCH,
                 micro_cell=0.1):
        self.corner = corner
        self.cell_shape = cell_shpae  # is this useful?
        self.data = data
        self.clustering_results = None
        self.clustering_type = clustering_type
        self.kernel_bandwidth = kernel_bandwidth
        self.cell_type = cell_type

        # special fields for streaming learning
        self.corners = []
        self.sums = []
        self.count = []
        self.cell_resolution = micro_cell

    def get_cell_centers(self, point):

        corner = np.round(point / self.cell_resolution, decimals=0) * self.cell_resolution - np.round(
            np.array([self.cell_resolution / 2, self.cell_resolution / 2]), decimals=0)
        return (corner[0], corner[1])

    def update(self, data):
        if self.cell_type == CLCellType.STREAM:
            l_data=data[['velocity', 'motion_angle']].to_numpy()
            for d in l_data:
                # get cell center
                cell_center = self.get_cell_centers(d)
                if cell_center in self.corners:
                    index = self.corners.index(cell_center)
                    self.sums[index] = self.sums[index] + d
                    self.count[index] = self.count[index] + 1
                else:
                    self.corners.append(cell_center)
                    self.sums.append(d)
                    self.count.append(1)

    def query(self):

        mean_shifter = ms.MeanShift()
        if self.cell_type is CLCellType.STREAM:
            cell_means = np.array(self.sums) / np.array(self.count)[:,None]
            self.clustering_results = mean_shifter.cluster(cell_means,
                                                           kernel_bandwidth=self.kernel_bandwidth)
        elif self.cell_type is CLCellType.BATCH:
            self.clustering_results = mean_shifter.cluster(self.data[['velocity', 'motion_angle']].to_numpy(),
                                                           kernel_bandwidth=self.kernel_bandwidth)
        else:
            print("Unknown clustering type")


class CLMap:
    def __init__(self, pool_num=-1):
        self.grid_step = None
        self.grid_radius = None
        self.grid_type = None
        self.grid_precision = None
        self.data = pd.DataFrame()
        self.cells_data = []
        self.clustering_type = ClusteringType.MS
        self.data_extent = None
        self.data_extent = {'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None}
        self.processing_type = None
        self.initial = True
        self.p_array = []
        self.total_number_of_observations = 0
        if pool_num == -1:
            self.pool_num = mp.cpu_count()
        else:
            self.pool_num = pool_num

    def get_cell_corner_in_dimension(self, coord):
        return np.round(coord / self.grid_step, decimals=0) * self.grid_step - np.round(self.grid_step / 2, decimals=0)

    def get_cell_corner(self, point):
        corner = np.round(point / self.grid_step, decimals=0) * self.grid_step - np.round(
            np.array([self.grid_step / 2, self.grid_step / 2]), decimals=0)
        return (corner[0], corner[1])

    def set_up_map(self, **kwargs):
        self.grid_step = kwargs.get('step', 1)
        self.grid_radius = kwargs.get('radius', 1)
        self.grid_type = kwargs.get('type', CLCellShape.SQUARE)
        self.grid_precision = kwargs.get('precision', 2)
        self.processing_type = kwargs.get('processing', CLCellType.BATCH)

        # add column for future discretisation according to CLCellType
        if self.grid_type == CLCellShape.CIRCULAR:
            pass
        if self.grid_type == CLCellShape.SQUARE:
            self.data["corners"] = ""

    def load_data(self, data):
        self.total_number_of_observations = len(data["time"].unique())

        if self.processing_type is CLCellType.BATCH:
            self.data = data
            self.data_extent['x_min'] = self.get_cell_corner_in_dimension(self.data['x'].min())
            self.data_extent['x_max'] = self.get_cell_corner_in_dimension(self.data['x'].max())
            self.data_extent['y_min'] = self.get_cell_corner_in_dimension(self.data['y'].min())
            self.data_extent['y_max'] = self.get_cell_corner_in_dimension(self.data['y'].max())
            if self.grid_type == CLCellShape.CIRCULAR:
                pass
            if self.grid_type == CLCellShape.SQUARE:
                self.data['corners'] = self.data.apply(lambda row: self.get_cell_corner(np.array([row['x'], row['y']])),
                                                       axis=1)
                cells = self.data['corners'].unique()
                for cell_data in cells:
                    cell = CLCell(cell_data, self.data.loc[self.data['corners'] == cell_data], self.clustering_type,
                                  self.grid_type)
                    self.cells_data.append(cell)
        elif self.processing_type is CLCellType.STREAM:
            if self.initial:
                self.data_extent['x_min'] = self.get_cell_corner_in_dimension(data['x'].min())
                self.data_extent['x_max'] = self.get_cell_corner_in_dimension(data['x'].max())
                self.data_extent['y_min'] = self.get_cell_corner_in_dimension(data['y'].min())
                self.data_extent['y_max'] = self.get_cell_corner_in_dimension(data['y'].max())
                self.initial = False
            else:
                self.data_extent['x_min'] = self.get_cell_corner_in_dimension(data['x'].min()) if self.data_extent[
                                                                                                      'x_min'] < self.get_cell_corner_in_dimension(
                    data['x'].min()) else self.get_cell_corner_in_dimension(data['x'].min())
                self.data_extent['x_max'] = self.get_cell_corner_in_dimension(data['x'].max()) if self.data_extent[
                                                                                                      'x_max'] > self.get_cell_corner_in_dimension(
                    data['x'].max()) else self.get_cell_corner_in_dimension(data['x'].max())
                self.data_extent['y_min'] = self.get_cell_corner_in_dimension(data['y'].min()) if self.data_extent[
                                                                                                      'y_min'] < self.get_cell_corner_in_dimension(
                    data['y'].min()) else self.get_cell_corner_in_dimension(data['y'].min())
                self.data_extent['y_max'] = self.get_cell_corner_in_dimension(data['y'].max()) if self.data_extent[
                                                                                                      'y_max'] > self.get_cell_corner_in_dimension(
                    data['y'].max()) else self.get_cell_corner_in_dimension(data['y'].max())

            if self.grid_type == CLCellShape.CIRCULAR:
                pass
            if self.grid_type == CLCellShape.SQUARE:
                data['corners'] = data.apply(lambda row: self.get_cell_corner(np.array([row['x'], row['y']])),
                                             axis=1)
                cells = data['corners'].unique()
                for cell_data in cells:
                    cell_to_update = next((x for x in self.cells_data if x.corner == cell_data), None)
                    if cell_to_update is None:
                        cell = CLCell(cell_data, [], self.clustering_type, self.grid_type, 0.5, CLCellType.STREAM)
                        cell.update(data.loc[data['corners'] == cell_data])
                        self.cells_data.append(cell)
                    else:
                        cell_to_update.update(data.loc[data['corners'] == cell_data])

    def cluster_data(self):
        # if self.processing_type is CLCellType.BATCH:
        with mp.Pool(self.pool_num) as p:
            self.cells_data = list(
                tqdm(p.imap(cluster_worker, (obj for obj in self.cells_data)), total=len(self.cells_data)))

            # for obj in tqdm(self.cells_data):
            #     obj.cluster_points()
