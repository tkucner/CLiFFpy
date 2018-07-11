# The MIT License (MIT)
#
# Copyright (c) 2015 Matt Nedrich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import utils as ut
import cl_aritmetics as cla
import point_grouper as pg

# import point_grouper as pg
# import mean_shift_utils as ms_utils

MIN_DISTANCE = 0.000001


class MeanShift(object):
    def __init__(self, kernel=ut.gaussian_kernel, distance=cla.distance_wrap_2d):
        self.kernel = kernel
        self.distance = distance

    def cluster(self, points, kernel_bandwidth, iteration_callback=None):
        if(iteration_callback):
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0
        print("Working")
        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            print("Working")
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)

                dist = self.distance(p_new, p_new_start)

                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        point_grouper = pg.PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())
        return MeanShiftResult(points, shift_points, group_assignments)

    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        points = np.array(points)
        point_rep=np.tile(point, [len(points), 1])
        # numerator

        point_weights = self.kernel(self.distance(point_rep, points), kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point

        # ***************************************************************************
        # ** The above vectorized code is equivalent to the unrolled version below **
        # ***************************************************************************
        # shift_x = float(0)
        # shift_y = float(0)
        # scale_factor = float(0)
        # for p_temp in points:
        #     # numerator
        #     dist = ms_utils.euclidean_dist(point, p_temp)
        #     weight = self.kernel(dist, kernel_bandwidth)
        #     shift_x += p_temp[0] * weight
        #     shift_y += p_temp[1] * weight
        #     # denominator
        #     scale_factor += weight
        # shift_x = shift_x / scale_factor
        # shift_y = shift_y / scale_factor
        # return [shift_x, shift_y]


class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids