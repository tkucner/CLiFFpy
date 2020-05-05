import pandas as pd
from enum import Enum
import numpy as np
import mean_shift as ms
import multiprocessing as mp
from tqdm import tqdm

class CLCellType(Enum):
    CIRCULAR = 1
    SQUARE = 2


class ClusteringType(Enum):
    MS = 1

def cluster_worker(obj):
    obj.cluster_points()
    return obj

class CLCell:
    def __init__(self, corner, data, clustering_type, cell_type, kernel_bandwidth=0.5):
        self.corner = corner
        self.cell_type = cell_type  # is this useful?
        self.data = data
        self.clustering_results = None
        self.clustering_type = clustering_type
        self.kernel_bandwidth = kernel_bandwidth
        self.mean_shift_result = None

    def cluster_points(self):
        mean_shifter = ms.MeanShift()
        self.mean_shift_result = mean_shifter.cluster(self.data[['velocity', 'motion_angle']].to_numpy(),
                                                      kernel_bandwidth=self.kernel_bandwidth)


class CLMap:
    def __init__(self, pool_num=-1):
        self.grid_step = None
        self.grid_radius = None
        self.grid_type = None
        self.grid_precision = None
        self.data_path = None
        self.data = pd.DataFrame()
        self.cells_data = []
        self.clustering_type = ClusteringType.MS
        self.data_extent = None
        self.data_extent = {'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None}
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

    def compute_data_extend(self):
        self.data_extent['x_min'] = self.get_cell_corner_in_dimension(self.data['x'].min())
        self.data_extent['x_max'] = self.get_cell_corner_in_dimension(self.data['x'].max())
        self.data_extent['y_min'] = self.get_cell_corner_in_dimension(self.data['y'].min())
        self.data_extent['y_max'] = self.get_cell_corner_in_dimension(self.data['y'].max())

    def set_up_map(self, **kwargs):
        self.grid_step = kwargs.get('step', 1)
        self.grid_radius = kwargs.get('radius', 1)
        self.grid_type = kwargs.get('type', CLCellType.SQUARE)
        self.grid_precision = kwargs.get('precision', 2)

        # add column for future discretisation according to CLCellType
        if self.grid_type == CLCellType.CIRCULAR:
            pass
        if self.grid_type == CLCellType.SQUARE:

            self.data["corners"] = ""

    def set_data_path(self, data_path):
        self.data_path = data_path

    def load_data(self, num_lines):
        self.data = pd.read_csv(self.data_path, nrows=num_lines)

    def split_data(self):
        if self.grid_type == CLCellType.CIRCULAR:
            pass
        if self.grid_type == CLCellType.SQUARE:
            self.data['corners'] = self.data.apply(lambda row: self.get_cell_corner(np.array([row['x'], row['y']])),
                                                   axis=1)
            cells = self.data['corners'].unique()
            for cell_data in cells:
                cell = CLCell(cell_data, self.data.loc[self.data['corners'] == cell_data], self.clustering_type,
                              self.grid_type)
                self.cells_data.append(cell)

    def cluster_data(self):
        with mp.Pool(self.pool_num) as p:
            self.cells_data = list(tqdm(p.imap(cluster_worker, (obj for obj in self.cells_data)), total=len(self.cells_data)))
        # for obj in tqdm(self.cells_data):
        #     obj.cluster_points()

