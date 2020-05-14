from cl_map import CLMap as cl
from cl_map_visualisation import CLVisualisation as clv
import pandas as pd



def logic(index):
    if index % 500 == 0 or index == 0:
        return False
    return True

# line format
# time [ms] (unixtime + milliseconds/1000), person id, position x [mm], position y [mm], position z (height) [mm], velocity [mm/s], angle of motion [rad], facing angle [rad]
# time, person_id, x, y, z, velocity, motion_angle, facing_angle

file_name = "/home/tzkr/Data/Directional/ATC/atc-20121114.csv"

cl_map = cl()

cl_map.set_up_map(step=1000)

cl_map.load_data(pd.read_csv(file_name, skiprows= lambda x: logic(x) ))

cl_map.cluster_data()

# visualisation
cl_vis = clv(cl_map)
cl_vis.show_raw_locations()
cl_vis.show_discretised_locations()
cl_vis.show_directions()
