import math
infinity = 0.00000001


class Line(object):
    def __init__(self, ):
        self.coefficient = None
        self.bias = None
        self.index = 0

    def two_points(self, shape):
        h, w = shape
        if self.coefficient >= 0:
            pt0 = self.get_point_by_x(0)
            pt1 = self.get_point_by_x(w)
        else:
            pt0 = self.get_point_by_y(0)
            pt1 = self.get_point_by_y(h)
        return [pt0, pt1]

    def set_coefficient_by_rotation(self, rotation):
        self.coefficient = math.tan(math.radians(rotation))

    def init_by_point_pair(self, pt0, pt1):
        x0, y0 = pt0
        x1, y1 = pt1
        # y0, x0 = pt0
        # y1, x1 = pt1
        if x1 > x0:
            self.coefficient = (y1 - y0) / (x1 - x0)
        elif x1 == x0:
            self.coefficient = (y0 - y1) / infinity
        else:
            self.coefficient = (y0 - y1) / (x0 - x1)
        self.bias = y0 - self.coefficient * x0

    def init_by_point_k(self, pt0, k):
        x0, y0 = pt0
        self.coefficient = k
        self.bias = y0 - k * x0

    def calculate_angle(self, ):
        angle = math.degrees(math.atan(self.coefficient))
        # if angle < 0:
        #     angle += 90
        # elif angle > 90:
        #     angle = 180 - angle
        return angle

    def get_point_by_x(self, x):
        return [x, self.coefficient * x + self.bias]

    def get_point_by_y(self, y):
        return [(y - self.bias) / self.coefficient, y]