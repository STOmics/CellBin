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

""" Get overlap&offset value of the nearest neighbor FOV."""
import os.path
import cv2
import tifffile
import numpy as np
from abc import ABC
import glog
from rich.progress import track
from .matcher import FFTMatcher, SIFTMatcher


# class FiledOfView(object):
#     def __init__(self):
#         self.height = None
#         self.width = None
#         self.channel = None
#         self.dtype = None
#         self.location = None
#         self.row_index = self.col_index = None
#
#     def set_index(self, r_ind, c_ind):
#         self.row_index = r_ind
#         self.col_index = c_ind
#
#     def set_location(self, loc): self.location = loc
#
#     def capture(self, url: str):
#         self.height = None
#         self.width = None
#         self.channel = None
#         self.dtype = None
#
#
# class TilesScanner(ABC):
#     def __init__(self, overlap=0.1):
#         self.overlap = overlap
#         self.rows = self.cols = self.count = None
#         self.canvas_width = self.canvas_height = None
#         self.fov = FiledOfView()
#         self.matcher = FFTMatcher()
#         self.matcher.overlap = overlap + 0.02
#
#     def scan(self, src: dict):
#         self._grid(src)
#
#         # raster in horizontal
#         self.matcher.horizontal = True
#
#         # raster in vertical
#         self.matcher.horizontal = False
#
#     def _grid(self, src: dict):
#         keys = np.array([k.split('_') for k in src.keys()], dtype=np.int)
#         x_end = np.max(keys[:, 0])
#         y_end = np.max(keys[:, 1])
#         x_start = np.min(keys[:, 0])
#         y_start = np.min(keys[:, 1])
#         self.rows = x_end - x_start + 1
#         self.cols = y_end - y_start + 1
#         glog.info('Raster in XY({}, {}) grid, X range from {} to {}, Y range from {} to {}.'.format(
#             self.rows, self.cols, x_start, x_end, y_start, y_end))


class TilesScanner(ABC):
    init_value = 999

    def __init__(self, overlap=0.10):
        self.rows = self.cols = None
        self.fov_height = self.fov_width = self.fov_channel = self.fov_dtype = None
        self.fov_count = None
        self.fov_overlap_array = None
        self.mosaic = None
        self.overlap = overlap
        self.horizontal_jitter = self.vertical_jitter = None
        self.data_pool = None
        self.fov_mask = None

        # Need output
        self.fov_loc_array = None
        self.mosaic_height = self.mosaic_width = None
        self.stitching_score = None
        self.transform_template = None
        self.template_src = None
        self.file_output: str = None

        self.matcher = FFTMatcher()

    def _row_col(self, keys):
        imgs_name_arr = list()
        for fov in keys:
            row, col = fov.split('_')
            imgs_name_arr.append([int(row), int(col)])
        imgs_name_arr = np.array(imgs_name_arr)
        fov_rows1 = np.max(imgs_name_arr[:, 0])
        fov_cols1 = np.max(imgs_name_arr[:, 1])
        rows_start = np.min(imgs_name_arr[:, 0])
        cols_start = np.min(imgs_name_arr[:, 1])
        self.rows = fov_rows1 - rows_start + 1
        self.cols = fov_cols1 - cols_start + 1

    def _init(self, pool):
        """ pool: dict or h5py.group """
        self.data_pool = pool
        r, c = list(self.data_pool.keys())[0].split('_')
        arr = self._get_image(int(r), int(c))
        self.fov_count = len(self.data_pool.keys())
        try:
            self.fov_height, self.fov_width, self.fov_channel = arr.shape
        except:
            self.fov_channel = 1
            self.fov_height, self.fov_width = arr.shape
        self.fov_dtype = arr.dtype
        # self.fov_count = len(self.data_pool.keys())
        self._row_col(self.data_pool.keys())
        glog.info('Total {} FOV in a grid(RC)[{} x {}], each FOV(Channel, Height, Width, type)=={}, {}, {}, {}.'.format(
            self.fov_count, self.rows, self.cols, self.fov_channel, self.fov_height, self.fov_width, self.fov_dtype))

    def _get_image(self, row: int, col: int, stitch_channel=None):
        """
        row: fov in row
        col: fov in col
        stitch_channel: if fov channel more than 1, can appoint channel
        """
        key = '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))
        if type(self.data_pool) is dict:
            arr = tifffile.imread(self.data_pool[key])
            if arr.ndim == 3:
                if arr.shape[0] == 2 or arr.shape[0] == 3:
                    arr = arr.transpose(1, 2, 0)
                if stitch_channel is not None:
                    assert stitch_channel <= arr.shape[2] and isinstance(stitch_channel, int), \
                        "stitch_channel ({}) is error".format(stitch_channel)
                    arr = arr[:, :, stitch_channel]
            return arr
        else:
            if key not in self.data_pool.keys():
                return None
            arr = self.data_pool[key][:]  # h5py.group -> dataset
            if arr.ndim == 3:
                if arr.shape[0] == 2 or arr.shape[0] == 3:
                    arr = arr.transpose(1, 2, 0)
                if stitch_channel is not None:
                    assert stitch_channel <= arr.shape[2] and isinstance(stitch_channel, int), \
                        "stitch_channel ({}) is error".format(stitch_channel)
                    arr = arr[:, :, stitch_channel]
            return arr

    def _set_mosaic_shape(self, ):
        x0 = np.min(self.fov_loc_array[:, :, 0])
        y0 = np.min(self.fov_loc_array[:, :, 1])
        x1 = np.max(self.fov_loc_array[:, :, 0])
        y1 = np.max(self.fov_loc_array[:, :, 1])
        w = x1 - x0 + self.fov_width
        h = y1 - y0 + self.fov_height
        # self.mosaic_shape = (int(h), int(w))
        self.mosaic_height = int(h)
        self.mosaic_width = int(w)
        glog.info('Mosaic size(WH) is {}, {}.'.format(self.mosaic_width, self.mosaic_height))

    def export_loc(self, file_path):
        np.save(os.path.join(file_path, 'location_horizontal_relative.npy'), self.horizontal_jitter)
        np.save(os.path.join(file_path, 'location_vertical_relative.npy'), self.vertical_jitter)
        np.save(os.path.join(file_path, 'location_global.npy'), self.fov_loc_array)

    def update_mosaic(self, ):
        """ output should be a file, stitched image. """
        self._set_mosaic_shape()
        mosaic_shape = (self.fov_channel == 1) and (self.mosaic_height, self.mosaic_width) or \
                       (self.mosaic_height, self.mosaic_width, self.fov_channel)
        self.mosaic = np.zeros(mosaic_shape, dtype=self.fov_dtype)
        for i in track(range(self.rows), description='Generating mosaic'):
            for j in range(self.cols):
                fov = self._get_image(i, j)
                x0, y0 = self.fov_loc_array[i, j]
                if self.fov_channel == 1:
                    self.mosaic[y0: y0 + self.fov_height, x0: x0 + self.fov_width] = fov
                else:
                    self.mosaic[y0: y0 + self.fov_height, x0: x0 + self.fov_width, :] = fov
                # TODO: blend & balance
        glog.warn('No blend processing for the mosaic.')

    def save_mosaic(self, output_path):
        if self.fov_channel == 1: arr = self.mosaic
        else: arr = self.mosaic.transpose(2, 0, 1)
        if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path))
        glog.info('Export mosaic to {}.'.format(output_path))
        tifffile.imwrite(output_path, arr)

    def set_matcher(self, mode):
        assert mode in ['FFT', 'SIFT']
        if mode == 'SIFT':
            self.matcher = SIFTMatcher()
        else:
            self.matcher = FFTMatcher()
            # self.matcher.overlap += 0.02

    def _create_jitter_tabel(self, stitch_channel=None):

        self.horizontal_jitter = np.zeros((self.rows, self.cols, 2), dtype=int) + self.init_value
        self.vertical_jitter = np.zeros((self.rows, self.cols, 2), dtype=int) + self.init_value
        self.fov_mask = np.zeros((self.rows, self.cols, 3))  # H,V,Tissue
        confi_mask = np.zeros((self.rows, self.cols, 2)) - 1
        ncc_confi_mask = np.zeros((self.rows, self.cols, 2)) - 1

        glog.info('Scan Row by Row,')
        for i in track(range(self.rows), description='RowByRow'):
            train = self._get_image(i, 0, stitch_channel)
            for j in range(1, self.cols):
                query = self._get_image(i, j, stitch_channel)
                if (train is not None) and (query is not None):
                    b = self.matcher.neighbor_match(train, query)
                    if b is not None:
                        self.horizontal_jitter[i, j, :] = b[:2]
                        confi_mask[i, j, 0] = b[2]
                        ncc_confi_mask[i, j, 0] = b[3]
                train = query
        self.horizontal_jitter, confi_mask[:, :, 0], h_offset_max = self.filter_abnormal_offset(self.horizontal_jitter,
                                                                                                confi_mask[:, :, 0],
                                                                                                thread=2)

        glog.info('Scan Col by Col,')
        self.matcher.horizontal = False
        for j in track(range(self.cols), description='ColByCol'):
            train = self._get_image(0, j, stitch_channel)
            for i in range(1, self.rows):
                query = self._get_image(i, j, stitch_channel)
                if (train is not None) and (query is not None):
                    b = self.matcher.neighbor_match(train, query)
                    if b is not None:
                        self.vertical_jitter[i, j, :] = b[:2]
                        confi_mask[i, j, 1] = b[2]
                        ncc_confi_mask[i, j, 1] = b[3]
                train = query
        self.vertical_jitter, confi_mask[:, :, 1], v_offset_max = self.filter_abnormal_offset(self.vertical_jitter,
                                                                                              confi_mask[:, :, 1],
                                                                                              thread=2)

        # confi_mask[:,:,0][np.where((self.horizontal_jitter[:,:,0]==self.init_value) & (confi_mask[:,:,0]!=-1))]=0
        # confi_mask[:,:,1][np.where((self.vertical_jitter[:,:,0]==self.init_value) & (confi_mask[:,:,1]!=-1))]=0

        # self.horizontal_jitter = self.horizontal_jitter - np.array([self.fov_width * self.overlap, 0])
        # self.vertical_jitter = self.vertical_jitter - np.array([0,self.fov_height*self.overlap])

        self.fov_mask[:, :, :2] = confi_mask
        self.fov_mask[:, :, 2] = np.max(confi_mask, axis=2)
        self.fov_mask[:, :, 2][np.where(self.fov_mask[:, :, 2] >= 99)] = 99

        del train, query

    def _fix_jitter(self, ):
        h_x = np.mean([self.horizontal_jitter[i, j, 0] for i in range(self.rows)
                       for j in range(self.cols) if self.horizontal_jitter[i, j, 0] != self.init_value])
        h_y = np.mean([self.horizontal_jitter[i, j, 1] for i in range(self.rows)
                       for j in range(self.cols) if self.horizontal_jitter[i, j, 1] != self.init_value])

        v_x = np.mean([self.vertical_jitter[i, j, 0] for i in range(self.rows)
                       for j in range(self.cols) if self.vertical_jitter[i, j, 0] != self.init_value])
        v_y = np.mean([self.vertical_jitter[i, j, 1] for i in range(self.rows)
                       for j in range(self.cols) if self.vertical_jitter[i, j, 1] != self.init_value])
        # h_x = h_x if h_x<999 else self.fov_width*self.overlap
        # h_y = h_y if h_y<999 else 0
        # v_x = v_x if v_x < 999 else self.fov_height * self.overlap
        # v_y = v_y if v_y < 999 else 0

        for i in range(self.rows):
            for j in range(self.cols):

                if self.horizontal_jitter[i, j, 0] == self.init_value:
                    if j == 0:
                        self.horizontal_jitter[i, j] = [0, 0]
                    else:
                        self.horizontal_jitter[i, j] = [round(h_x), round(h_y)]

                if self.vertical_jitter[i, j, 0] == self.init_value:
                    if i == 0:
                        self.vertical_jitter[i, j] = [0, 0]
                    else:
                        self.vertical_jitter[i, j] = [round(v_x), round(v_y)]

    @staticmethod
    def check_tissue_fov(image):
        """check whether it is an organizational region"""
        image = cv2.GaussianBlur(image, (15, 15), 0)
        grad = cv2.Sobel(image, -1, 1, 1)
        return int(grad.mean())

    # @staticmethod
    # def filter_abnormal_offset(offset: np.array,mask):
    #     """
    #      offset outlier filtering
    #     :param offset: W*H*C
    #     :return:
    #     """
    #     c = offset.shape[2]
    #     max_thread = 0
    #     for i in range(c):
    #         offset_1 = offset[:, :, i]  # 所有offset
    #         valid_offset = offset_1[np.where((offset_1 != -999) & (mask == 1))]

    #         if len(valid_offset) > 3:
    #             theta = 3 if valid_offset.std() > 5 else 4
    #             valid_offset = np.delete(valid_offset, np.argmax(valid_offset))
    #             valid_offset = np.delete(valid_offset, np.argmin(valid_offset))
    #
    #             thread_max = valid_offset.mean() + valid_offset.std() * theta
    #             thread_min = valid_offset.mean() - valid_offset.std() * theta
    #             offset[np.where(offset_1 > thread_max)] = 999
    #             offset[np.where(offset_1 < thread_min)] = 999
    #             if thread_max > max_thread:
    #                 max_thread = thread_max
    #     return offset, max_thread

    @staticmethod
    def filter_abnormal_offset(offset: np.array, stitch_confi_mask: np.ndarray, thread: int=2):
        """
        outlier filtering offset
        :param offset: W*H*C
        :return:
        """
        c = offset.shape[2]
        max_thread = 0
        for i in range(c):
            offset_1 = offset[:, :, i]  # all offset
            valid_offset = offset_1[np.where((offset_1 != -999) & (stitch_confi_mask > thread))]
            # Remove the maximum and minimum values and take out the value
            # between [0.2, 0.8] to find the mean and variance
            if len(valid_offset) > 3:
                percentile = np.percentile(valid_offset, (25, 50, 75), interpolation='midpoint')
                Q1, Q3 = percentile[0], percentile[2]
                IQR = Q3 - Q1
                IQR = IQR if IQR > 5 else IQR + 3

                thread_max = Q3 + 1.5 * IQR
                thread_min = Q1 - 1.5 * IQR

                error_fov = np.zeros(shape=(0, 2), dtype=np.int32)
                max_error_fov = np.array(np.where((offset_1 > thread_max))).T
                min_error_fov = np.array(np.where((offset_1 < thread_min) & (offset_1 != -999))).T
                error_fov = np.vstack([error_fov, max_error_fov, min_error_fov])
                if len(error_fov) > 0:
                    for row, col in error_fov:
                        if stitch_confi_mask[row, col] > thread * 3:  # High confidence, relaxed threshold
                            thread_max += IQR * 0.6
                            thread_min -= IQR * 0.6
                        if offset_1[row, col] > thread_max or offset_1[row, col] < thread_min:
                            offset[row, col] = -999
                            stitch_confi_mask[row, col] = 0

                if thread_max > max_thread:
                    max_thread = thread_max
        return offset, stitch_confi_mask, max_thread

# s = Stitcher()
# s.load(r'D:\data\cellbin_all\motic\1_origin\SS200000171BL_D3')


def main(): pass


if __name__ == '__main__':
    main()
