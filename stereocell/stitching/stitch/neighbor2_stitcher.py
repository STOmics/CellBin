# Copyright 2022 Beijing Genomics Institute(BGI) Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Get FOV' location by overlap&offset."""

import numpy as np
from .tiles_scanner import TilesScanner


class Neighbor2(TilesScanner):
    def __init__(self, overlap: float = 0.12):
        super().__init__(overlap)
        self._support = ['.tif', '.png', '.jpg']

    def stitching(self, src_data):
        """ Just stitch for raw fov """
        self.set_matcher("FFT")
        self._init(src_data)
        self._create_jitter_tabel()
        self._fix_jitter()
        self._custom_stitch()
        self.update_mosaic()

    def _custom_stitch(self, ): 
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
