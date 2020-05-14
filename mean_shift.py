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
import cl_arithmetic as cla
import point_grouper as pg

MIN_DISTANCE = 0.000001


class MeanShift(object):
    def __init__(self, kernel=ut.gaussian_kernel, distance=cla.distance_wrap_2d_vec, weight=cla.weighted_mean_2d_vec):
        self.kernel = kernel
        self.distance = distance
        self.weight = weight

    def cluster(self, points, kernel_bandwidth, iteration_callback=None):
        if iteration_callback:
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        history = points
        history = history.tolist()
        for i in range(0, len(history)):
            history[i] = [history[i]]

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)

                dist = self.distance(p_new, p_new_start)

                history[i].append(p_new)
                # print(history[i])

                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        point_grouper = pg.PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())

        return MeanShiftResult(points, shift_points, group_assignments, history)

    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        points = np.array(points)
        point_rep = np.tile(point, [len(points), 1])
        dist = self.distance(point_rep, points)
        point_weights = self.kernel(dist, kernel_bandwidth)

        shifted_point = self.weight(points, point_weights)
        return shifted_point


class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids, history):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids
        self.history = history
        self.mixing_factors = []
        self.covariances = []
        self.mean_values = []
        # compute GMM parameters
        unique_cluster_ids, counts = np.unique(self.cluster_ids, return_counts=True)
        for uid, c in zip(unique_cluster_ids, counts):
            self.mixing_factors.append(c / self.cluster_ids.size)
            self.mean_values.append(np.mean(self.original_points[self.cluster_ids == uid, :], axis=0))
            self.covariances.append(np.cov(self.original_points[self.cluster_ids == uid, :]))
