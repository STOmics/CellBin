import numpy as np


class SpanningTree(object):
    def __init__(self):
        self.jitter: dict = {}
        self.fov_loc_array = None
        self.fov_height = self.fov_width = None
        self.cols = self.rows = 0

    def generate(self, mode=''):
        self.fov_loc_array = np.zeros((self.rows, self.cols, 2), dtype=int)
        for i in range(0, self.rows):
            info = np.zeros((self.cols, 2), dtype=int)
            for j in range(1, self.cols):
                info[j, :] = info[j - 1, :] + self.horizontal_jitter[i, j, :] + [self.fov_width, 0]
            if i == 0: delta = [0, 0]
            else:
                delta = self.fov_loc_array[i - 1, :, :] + \
                        [0, self.fov_height] - info + self.vertical_jitter[i, :, :]
            self.fov_loc_array[i, :, :] = info + delta
        self.fov_loc_array[:, :, 0] -= np.min(self.fov_loc_array[:, :, 0])
        self.fov_loc_array[:, :, 1] -= np.min(self.fov_loc_array[:, :, 1])
