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
from sklearn.linear_model import LinearRegression
import scipy.spatial as spt


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
        cd = CenterLrDiffusionStitch(self.rows, self.cols)

        self.horizontal_jitter[np.where(self.horizontal_jitter == -999)] = 999
        self.vertical_jitter[np.where(self.vertical_jitter == -999)] = 999
        cd.setJitter(self.horizontal_jitter, self.vertical_jitter)
        cd.set_scope_loc_by_overlap(self.fov_height, self.fov_width, self.overlap)
        cd.multi_connect_domain_center()
        cd.multi_center_2_global_loc()
        self.fov_loc_array = cd.global_loc
        self.horizontal_jitter, _, self.vertical_jitter, _ = self.global_loc_to_offset(self.fov_loc_array)

    def global_loc_to_offset(self, stitched_loc: np.ndarray):
        """
        The deviation of adjacent fov was calculated according to the mosaic coordinate or overlap of the microscope,
        and then the deviation from the calculation result of fft was calculated
        stitched_loc(N*M*2): global location of scope stitching
        return: horizontal_offset,vertical_offset,stitched_loc
        """
        horizontal_scope = np.zeros_like(stitched_loc, dtype=np.int32)
        vertical_scope = np.zeros_like(stitched_loc, dtype=np.int32)
        rows, cols, c = stitched_loc.shape

        horizontal_scope[:, 1:] = (stitched_loc[:, 1:] - stitched_loc[:, :-1])
        horizontal_scope[:, 1:, 0] -= self.fov_width
        horizontal_confi = np.zeros(shape=(rows, cols))
        horizontal_confi[:, 1:] = 10

        vertical_scope[1:] = (stitched_loc[1:] - stitched_loc[:-1])
        vertical_scope[1:, :, 1] -= self.fov_height
        vertical_confi = np.zeros(shape=(rows, cols))
        vertical_confi[1:] = 10

        return horizontal_scope, horizontal_confi, vertical_scope, vertical_confi


class CenterLrDiffusionStitch:
    '''
        * * *    * # *    # # #
        * # * -> # # # -> # # #
        * * *    * # *    # # #
    '''

    def __init__(self, rows, cols):
        '''
        rows, cols:
        src: (row, col)
        '''
        self.scope_global_loc = None
        self.rows = rows
        self.cols = cols
        self.fov_height = None
        self.fov_width = None
        # self.src = src

        self.jitter_mask = np.zeros((self.rows, self.cols), dtype=int) + 999  # connected domain information
        self.stitch_mask = np.zeros((self.rows, self.cols), dtype=int)  # index to be stitched
        self.stitch_masked = np.zeros((self.rows, self.cols), dtype=int)  # concatenated index
        self.global_loc = np.zeros((self.rows, self.cols, 2), dtype=int) + 999
        self.stitch_list = list()  # final stitching order
        self.connect_domains = dict()  # connected domain information
        self.domains = -1  # connected domain number

    def setJitter(self, h_j, v_j):
        '''set jitter matrix'''
        assert h_j.shape == v_j.shape, "jitter ndim is diffient"
        self.horizontal_jitter = h_j
        self.vertical_jitter = v_j

    def set_scope_loc(self, scope_loc: np.ndarray):
        """set scope loction"""
        self.scope_global_loc = scope_loc

    def set_fov_size(self, fov_height, fov_width):
        """set fov size"""
        self.fov_height = fov_height
        self.fov_width = fov_width

    def set_scope_loc_by_overlap(self, h, w, overlap):
        """ Compute the raw stitching coordinates of the microscope """
        scope_global_loc = np.zeros(shape=(self.rows, self.cols, 2), dtype=np.int32)
        self.fov_height = h
        self.fov_width = w
        for i in range(self.rows):
            for j in range(self.cols):
                scope_global_loc[i, j, 0] = j * (w - int(w * overlap))
                scope_global_loc[i, j, 1] = i * (h - int(h * overlap))
        self.scope_global_loc = scope_global_loc

    def neighbor(self, row, col, mask: np.ndarray = None):
        """
        Gets the unstitching FOV of the specified column position attachment
        """
        if (col - 1) < 0:
            left = None
        else:
            left = (row, col - 1)
        if (col + 1) > (self.cols - 1):
            right = None
        else:
            right = (row, col + 1)
        if (row - 1) < 0:
            up = None
        else:
            up = (row - 1, col)
        if (row + 1) > (self.rows - 1):
            down = None
        else:
            down = (row + 1, col)

        if mask is None:
            return [left, right, up, down]
        else:
            mask_list = list()
            for it in [left, right, up, down]:
                if it and mask[it[0], it[1]] == 1:
                    mask_list.append(it)
            return mask_list

    def neighbor_fix(self, row, col):
        """
        Gets the unstitching FOV of the specified column position attachment
        :return: [r, c]
        """
        if (col - 1) < 0:
            left = None
        else:
            left = [row, col - 1]
        if (col + 1) > (self.cols - 1):
            right = None
        else:
            right = [row, col + 1]
        if (row - 1) < 0:
            up = None
        else:
            up = [row - 1, col]
        if (row + 1) > (self.rows - 1):
            down = None
        else:
            down = [row + 1, col]

        for it in [left, right, up, down]:
            if it:
                if self.stitch_masked[it[0], it[1]] == 1:
                    return it
        return None

    def _getFirstPosition(self, r=True):
        ''' Find the starting point of the connected domain
        r: Control whether to add connected domain names
        '''
        for i in range(self.rows):
            for j in range(self.cols):
                if self.horizontal_jitter[i, j, 0] != 999 or \
                        self.vertical_jitter[i, j, 0] != 999:
                    if (i, j) not in self.connect_domains.keys():
                        if r: self.domains += 1
                        return (i, j)
        return None

    def _indexIsLegal(self, index):
        ''' Determine whether the row or column number is legal '''
        row, col = index
        if 0 <= row < self.rows and \
                0 <= col < self.cols:
            return True
        return False

    def caculateCenter(self, dst=None):
        ''' Find connected domains recursively '''
        h_flag = False
        new_dst = list()
        if dst is None:
            dst = [self._getFirstPosition()]

        for (row, col) in dst:
            # if (row, col) not in self.connect_domains.keys():
            self.connect_domains[(row, col)] = self.domains
            self.jitter_mask[row, col] = self.domains
            # temp = self.neighbor(row, col)

            if self.horizontal_jitter[row, col, 0] != 999:
                if self._indexIsLegal((row, col - 1)):
                    if (row, col - 1) not in self.connect_domains.keys():
                        self.connect_domains[(row, col - 1)] = self.domains
                        self.jitter_mask[row, col - 1] = self.domains
                        if (row, col - 1) not in new_dst:
                            new_dst.append((row, col - 1))
            if self.vertical_jitter[row, col, 0] != 999:
                if self._indexIsLegal((row - 1, col)):
                    if (row - 1, col) not in self.connect_domains.keys():
                        self.connect_domains[(row - 1, col)] = self.domains
                        self.jitter_mask[row - 1, col] = self.domains
                        if (row - 1, col) not in new_dst:
                            new_dst.append((row - 1, col))

            if self._indexIsLegal((row, col + 1)):
                if self.horizontal_jitter[row, col + 1, 0] != 999:
                    if (row, col + 1) not in self.connect_domains.keys():
                        if (row, col + 1) not in new_dst:
                            new_dst.append((row, col + 1))
            if self._indexIsLegal((row + 1, col)):
                if self.vertical_jitter[row + 1, col, 0] != 999:
                    if (row + 1, col) not in self.connect_domains.keys():
                        if (row + 1, col) not in new_dst:
                            new_dst.append((row + 1, col))

        if len(new_dst) != 0:
            self.caculateCenter(new_dst)
        else:
            return

    def caculateDomains(self):
        '''Find the connected domain multiple times until the end of the search
        return: jitter_mask, connect_domains
        '''
        while self._getFirstPosition(r=False) is not None:
            self.caculateCenter()

    def getStitchCenter(self, max_domain=None):
        ''' Obtain the center index of the largest connected domain splicing '''
        if max_domain is None:
            max_domain = max(self.connect_domains.values(), key=list(self.connect_domains.values()).count)
        else:
            assert max_domain in np.arange(0, self.domains + 1), 'max_domain input error'
        domains = np.array([i for i in self.connect_domains.keys() if self.connect_domains[i] == max_domain])
        max_row = np.max(domains[:, 0])
        min_row = np.min(domains[:, 0])
        max_col = np.max(domains[:, 1])
        min_col = np.min(domains[:, 1])
        center_row = int((max_row + min_row) / 2)
        center_col = int((max_col + min_col) / 2)
        return center_row, center_col

    def _getStitchOrder_by_domain(self, stitch_order):
        ''' get order '''
        domain_mask = np.zeros(shape=(self.rows, self.cols))
        stitch_list = []
        for key, value in self.connect_domains.items():
            if value == self.connect_domains[stitch_order[0]]:
                row, col = key
                domain_mask[row, col] = 1

        while len(stitch_order) > 0:
            index = stitch_order.pop(0)
            if index not in stitch_list:
                stitch_list.append(index)
                nei = self.neighbor(index[0], index[1], mask=domain_mask)
                for i in nei:
                    if i not in stitch_list and i is not None:
                        stitch_order.append(i)
        return stitch_list

    def _getStitchOrder(self, stitch_order):
        '''get order '''
        # domain_mask = np.zeros(shape=(self.rows,self.cols))
        stitch_list = []

        while len(stitch_order) > 0:
            index = stitch_order.pop(0)
            if index not in stitch_list:
                stitch_list.append(index)
                nei = self.neighbor(index[0], index[1])
                for i in nei:
                    if i not in stitch_list and i is not None:
                        stitch_order.append(i)
        return stitch_list

    def centerToGlobal(self, row=None, col=None):
        ''' Expand all stitching sequences from the center
        Support for external typing row, col
        Otherwise, calculate the center of the largest connected domain
        '''
        if row is None and col is None:
            self.caculateDomains()
            row, col = self.getStitchCenter()
            self.center_row = row
            self.center_col = col

        stitch_order = list()
        stitch_order.append((row, col))
        self.stitch_list = self._getStitchOrder(stitch_order)

    def domain_order(self, max_domain):
        """
        Starting from the specified connected domain, splicing the corresponding connected domains in order
        :param max_domain:
        :return:
        """
        first_row, first_col = self.getStitchCenter(max_domain)

        domain_order = [(first_col, first_col)]
        domain_list = []
        domain_pass_list = []
        while len(domain_order) > 0:
            index = domain_order.pop(0)
            if index not in domain_pass_list:
                domain_pass_list.append(index)
                nei = self.neighbor(index[0], index[1])
                for i in nei:
                    if i is not None:
                        domain_order.append(i)
                        domain = self.jitter_mask[i[0], i[1]]
                        if domain not in domain_list and domain != 999:
                            domain_list.append(domain)
        return domain_list

    def multi_connect_domain_center(self):
        ''' Expand all stitching sequences from the center
        Support for external typing row, col
        Otherwise, calculate the center of the largest connected domain
        '''

        # if row is None and col is None:
        self.caculateDomains()
        max_domain = max(self.connect_domains.values(), key=list(self.connect_domains.values()).count)
        self.domain_list = self.domain_order(max_domain)

    def multi_center_2_global_loc(self):
        self.lr = LinearRegression()
        self.lr.fit(np.array([[0, 0]]), np.array([[0, 0]]))
        if (self.lr.coef_ == 0).all():
            self.lr.coef_ = np.diag(np.ones(2))

        for domain_idx in self.domain_list:
            center_row, center_col = self.getStitchCenter(max_domain=domain_idx)

            stitch_order = list()
            stitch_order.append((center_row, center_col))
            stitch_list = self._getStitchOrder_by_domain(stitch_order)
            self.get_domain_Loc(self.fov_height, self.fov_width, stitch_list)

            self.lr.fit(self.scope_global_loc[np.where(self.stitch_masked == 1)],
                        self.global_loc[np.where(self.stitch_masked == 1)])

        self.fix_unstitch_loc()

        self.global_loc[:, :, 0] -= np.min(self.global_loc[:, :, 0])
        self.global_loc[:, :, 1] -= np.min(self.global_loc[:, :, 1])

    def _fixJitter(self, ):
        '''
        Complement the horizontal and vertical offset matrix
        '''
        h_x = np.mean([self.horizontal_jitter[i, j, 0] for i in range(self.rows)
                       for j in range(self.cols) if self.horizontal_jitter[i, j, 0] != 999])
        h_y = np.mean([self.horizontal_jitter[i, j, 1] for i in range(self.rows)
                       for j in range(self.cols) if self.horizontal_jitter[i, j, 1] != 999])

        v_x = np.mean([self.vertical_jitter[i, j, 0] for i in range(self.rows)
                       for j in range(self.cols) if self.vertical_jitter[i, j, 0] != 999])
        v_y = np.mean([self.vertical_jitter[i, j, 1] for i in range(self.rows)
                       for j in range(self.cols) if self.vertical_jitter[i, j, 1] != 999])

        for i in range(self.rows):
            for j in range(self.cols):
                if self.horizontal_jitter[i, j, 0] == 999:
                    if j == 0:
                        self.horizontal_jitter[i, j] = [0, 0]
                    else:
                        if self.jitter_mask[i, j] == 999:
                            self.horizontal_jitter[i, j] = [round(h_x), round(h_y)]

                if self.vertical_jitter[i, j, 0] == 999:
                    if i == 0:
                        self.vertical_jitter[i, j] = [0, 0]
                    else:
                        if self.jitter_mask[i, j] == 999:
                            self.vertical_jitter[i, j] = [round(v_x), round(v_y)]

    def get_nearest_points(self, input_point, template_points):
        # get the closest point
        tree = spt.cKDTree(data=template_points)
        distance, index = tree.query(input_point, k=2)
        neighbor_point = template_points[index[0]]
        distance = np.abs(input_point - neighbor_point)
        return distance, neighbor_point

    def get_nearest_domain_fov(self, current_domain, stitched_loc):
        """ Obtain the row and column number of the FOV closest to the spliced connected domain
        in the current connected domain """
        min_distance = 999
        nearst_domain = None
        for fov_loc in current_domain:
            distance, _ = self.get_nearest_points(fov_loc, stitched_loc)
            distance = np.linalg.norm(distance)
            if distance < min_distance:
                min_distance = distance
                nearst_domain = fov_loc
        return nearst_domain, min_distance

    def get_domain_Loc(self, height, width, stitch_list):
        ''' Get stitched coordinates of connected domains '''
        # self._fixJitter()
        stitch_mask = np.zeros(shape=(self.rows, self.cols))
        nearest_domain_fov = None
        for item in stitch_list:
            row, col = item
            self.stitch_masked[row, col] = 1
            if np.max(stitch_mask) == 0:
                # Find the closest FOV to the connected domain that has been spliced in the current domain
                stitched_loc = np.where(self.stitch_masked == 1)
                stitched_loc = np.vstack(stitched_loc).T
                nearest_domain_fov, _ = self.get_nearest_domain_fov(stitch_list, stitched_loc)
                stitch_mask[row, col] = 1
                self.global_loc[row, col] = self.lr.predict([self.scope_global_loc[row, col]])[0]
            else:
                mask_list = self.neighbor(row, col, mask=stitch_mask)

                loc_list = list()
                for mask in mask_list:
                    row_src, col_src = mask
                    temp_loc = None
                    if row_src > row:
                        if self.vertical_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([0, height] + self.vertical_jitter[row_src, col_src])
                    elif row_src < row:
                        if self.vertical_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([0, height] + self.vertical_jitter[row, col])
                    elif col_src > col:
                        if self.horizontal_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([width, 0] + self.horizontal_jitter[row_src, col_src])
                    elif col_src < col:
                        if self.horizontal_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([width, 0] + self.horizontal_jitter[row, col])
                    if temp_loc is not None:
                        loc_list.append(temp_loc)

                if len(loc_list) >= 1:
                    loc_list = np.array(loc_list)
                    self.global_loc[row, col] = np.mean(loc_list, axis=0) + 0.5
                    stitch_mask[row, col] = 1
                else:
                    stitch_list.append(item)
                    print("{}:{} no stitch neighbor".format(row, col))
        # Adjust the position of the FOV of the nearest neighbor connected domain to the predicted position
        if nearest_domain_fov is not None:
            nearest_fov_scope_loc = self.scope_global_loc[nearest_domain_fov[0], nearest_domain_fov[1]]
            offset = self.global_loc[nearest_domain_fov[0], nearest_domain_fov[1]] - \
                     self.lr.predict([nearest_fov_scope_loc])[0]
            offset = np.around(offset)
            self.global_loc[np.where(stitch_mask == 1)] -= offset.astype(np.int32)

    def getLastGlobalLoc(self, height, width):
        ''' Get the final splicing coordinates '''
        self._fixJitter()
        for item in self.stitch_list:
            row, col = item
            if np.max(self.stitch_mask) == 0:
                self.stitch_mask[row, col] = 1
                self.global_loc[row, col] = [0, 0]
            else:
                mask_list = self.neighbor(row, col, mask=self.stitch_mask)
                self.stitch_mask[row, col] = 1
                loc_list = list()
                for mask in mask_list:
                    row_src, col_src = mask
                    temp_loc = None
                    if row_src > row:
                        if self.vertical_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([0, height] + self.vertical_jitter[row_src, col_src])
                    elif row_src < row:
                        if self.vertical_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([0, height] + self.vertical_jitter[row, col])
                    elif col_src > col:
                        if self.horizontal_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([width, 0] + self.horizontal_jitter[row_src, col_src])
                    elif col_src < col:
                        if self.horizontal_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([width, 0] + self.horizontal_jitter[row, col])
                    if temp_loc is not None:
                        loc_list.append(temp_loc)

                if len(loc_list) >= 1:
                    loc_list = np.array(loc_list)
                    self.global_loc[row, col] = np.mean(loc_list, axis=0)
                else:
                    print("{}:{} no stitch neighbor".format(row, col))

        self.global_loc[:, :, 0] -= np.min(self.global_loc[:, :, 0])
        self.global_loc[:, :, 1] -= np.min(self.global_loc[:, :, 1])

    def check_feature_matrix(self, feature_mask):
        '''
        check feature mask to get neighbor
        :return:
        '''
        rows, cols = feature_mask.shape
        fix_list = []
        for row in range(rows):
            for col in range(cols):
                if self.stitch_masked[row, col] == 0:
                    it = self.neighbor_fix(row, col)
                    if it is not None:
                        fix_list.append(([row, col], it))
        return fix_list

    def check_up_stitch(self, row, col):
        '''
        Check whether the specified FOV can be stitched based on the offset from the algorithm
        :param row:
        :param col:
        :return:
        '''
        flag = False
        if self.stitch_masked[row, col] != 1 and row >= 0 and col >= 0:
            loc = []
            if col > 0 and self.horizontal_jitter[row, col, 0] != 999 and self.stitch_masked[row, col - 1] == 1:  # left
                x0, y0 = self.global_loc[row, col - 1] + [self.fov_width, 0] + \
                         self.horizontal_jitter[row, col]
                loc.append([x0, y0])
            if row < self.rows - 1 and self.vertical_jitter[row + 1, col, 0] != 999 and self.stitch_masked[
                row + 1, col] == 1:  # down
                x1, y1 = self.global_loc[row + 1, col] - ([0, self.fov_height] + self.vertical_jitter[row + 1, col])
                loc.append([x1, y1])
            if col < self.cols - 1 and self.horizontal_jitter[row, col + 1, 0] != 999 and self.stitch_masked[
                row, col + 1] == 1:  # right
                x2, y2 = self.global_loc[row, col + 1] - ([self.fov_width, 0] + self.horizontal_jitter[row, col + 1])
                loc.append([x2, y2])
            if row > 0 and self.vertical_jitter[row, col, 0] != 999 and self.stitch_masked[row - 1, col] == 1:  # up
                x3, y3 = self.global_loc[row - 1, col] + (
                        [0, self.fov_height] + self.vertical_jitter[row, col])
                loc.append([x3, y3])
            if len(loc) != 0:
                self.global_loc[row, col] = np.mean(loc, axis=0)
                self.stitch_masked[row, col] = 1
                flag = True
        return flag

    def fix_unstitch_loc(self):
        """
        The unstitching FOV is estimated to be stitching
        :return:
        """
        fix_list = self.check_feature_matrix(self.stitch_masked)
        h_mean_all = np.mean([self.horizontal_jitter[i, j, :] for i in range(self.rows) for j in range(self.cols)
                              if self.horizontal_jitter[i, j, 0] != 999], axis=0)
        v_mean_all = np.mean([self.vertical_jitter[i, j, :] for i in range(self.rows) for j in range(self.cols)
                              if self.vertical_jitter[i, j, 0] != 999], axis=0)
        while len(fix_list) > 0:
            for dst, src in fix_list:
                flag = self.check_up_stitch(dst[0], dst[1])
                if flag is False:
                    h_mean_list = [self.horizontal_jitter[row, dst[1], :] for row in range(self.rows)
                                   if self.horizontal_jitter[row, dst[1], 0] != 999]
                    h_mean = np.mean(h_mean_list, axis=0) if len(h_mean_list) > 0 else h_mean_all
                    v_mean_list = [self.vertical_jitter[dst[0], col, :] for col in range(self.cols)
                                   if self.vertical_jitter[dst[0], col, 0] != 999]
                    # Take out all the offsets in this column
                    v_mean = np.mean(v_mean_list, axis=0) if len(v_mean_list) > 0 else v_mean_all

                    # Find the neighbors that have been spelled around the current dst
                    neighbor = self.neighbor(dst[0], dst[1], self.stitch_masked)
                    tem_loc = []
                    for src in neighbor:
                        if dst[0] < src[0]:  # down
                            x0, y0 = self.global_loc[src[0], src[1]] - [0, self.fov_height] - v_mean
                            tem_loc.append([x0, y0])
                        elif dst[0] > src[0]:  # up
                            x0, y0 = self.global_loc[src[0], src[1]] + [0, self.fov_height] + v_mean
                            tem_loc.append([x0, y0])
                        elif dst[1] < src[1]:  # right
                            x0, y0 = self.global_loc[src[0], src[1]] - [self.fov_width, 0] - h_mean
                            tem_loc.append([x0, y0])
                        elif dst[1] > src[1]:  # left
                            x0, y0 = self.global_loc[src[0], src[1]] + [self.fov_width, 0] + h_mean
                            tem_loc.append([x0, y0])
                        else:
                            print('[{}-{}] fix_loc wrror!'.format(dst[0], dst[1]))
                    if len(tem_loc) > 0:
                        self.global_loc[dst[0], dst[1]] = np.mean(tem_loc, axis=0)
                        self.stitch_masked[dst[0], dst[1]] = 1

            fix_list = self.check_feature_matrix(self.stitch_masked)
        self.global_loc = np.around(self.global_loc, decimals=0).astype(np.int32)

        self.global_loc[:, :, 0] -= np.min(self.global_loc[:, :, 0])
        self.global_loc[:, :, 1] -= np.min(self.global_loc[:, :, 1])

