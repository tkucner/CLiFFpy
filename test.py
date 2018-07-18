import unittest
import cl_point
import cl_arithmetic


class TestDistanceMetrics(unittest.TestCase):
    def setUp(self):
        self.places = 4

        self.test_a = []
        self.test_wrap_a = []

        self.test_a.append(-1)
        self.test_wrap_a.append(-1)

        self.test_p1 = []
        self.test_p2 = []
        self.test_d_cos = []
        self.test_d_wrap = []

        self.test_p1.append(cl_point.CL2DPoint(1, 1, 1, 1))
        self.test_p2.append(cl_point.CL2DPoint(1, 1, 1, 2))
        self.test_d_cos.append(1)
        self.test_d_wrap.append(1)

        self.test_p1.append(cl_point.CL2DPoint(1, 1, 0, 1))
        self.test_p2.append(cl_point.CL2DPoint(1, 1, 1, 2))
        self.test_d_cos.append(1.4597)
        self.test_d_wrap.append(1.4142)

    def test_wrap_to_pi(self):
        for i in range(len(self.test_a)):
            self.assertAlmostEqual(cl_arithmetic.wrap_to_pi(self.test_a[i]), self.test_wrap_a[i], self.places)

    def test_distance_cos_2d(self):
        for i in range(len(self.test_d_cos)):
            self.assertAlmostEqual(cl_arithmetic.distance_cos_2d(self.test_p1[i], self.test_p2[i]), self.test_d_cos[i],
                                   self.places)

    def test_distance_wrap_2d(self):
        for i in range(len(self.test_d_cos)):
            self.assertAlmostEqual(cl_arithmetic.distance_wrap_2d(self.test_p1[i], self.test_p2[i]), self.test_d_wrap[i],
                                   self.places)


if __name__ == '__main__':
    unittest.main()