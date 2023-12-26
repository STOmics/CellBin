import glog
import numpy as np
import os
import time
from cellbin import CellBinElement
from cellbin.stitching.wsi_stitch import StitchingWSI
from cellbin import Image
from cellbin.stitching.fov_aligner import FOVAligner
from cellbin.stitching.global_location import GlobalLocation
from cellbin.stitching.template_reference import TemplateReference
from cellbin.utils.util import get_rc_from_image_map


class Stitcher(CellBinElement):
    def __init__(self):
        super(Stitcher, self).__init__()
        self.fov_location = None
        self.fov_x_jitter = None
        self.fov_y_jitter = None
        self.rows = self.cols = None
        self._fov_height = self._fov_width = self._fov_channel = None
        self._fov_dtype = None
        self.height = self.width = None
        self._overlap = 0.1

        self.__jitter_diff = None
        self.__offset_diff = None
        self.__template_max_value = None
        self.__template_mean_value = None
        self._template = None
        self.__image = None

    def get_mosaic(self, ): return self.__image

    def _init_parm(self, src_image: dict):
        test_image_path = list(src_image.values())[0]
        img = Image()
        img.read(test_image_path)

        self._fov_height = img.height
        self._fov_width = img.width
        self._fov_channel = img.channel
        self._fov_dtype = img.dtype
        r, c = get_rc_from_image_map(src_image)
        self.set_size(rows=r, cols=c)

    def set_size(self, rows, cols):
        assert isinstance(rows, int) or isinstance(rows, float), "Rows type error."
        assert isinstance(cols, int) or isinstance(cols, float), "Cols type error."
        self.rows = rows
        self.cols = cols

    def set_global_location(self, loc):
        assert type(loc) == np.ndarray, "Location type error."
        self.fov_location = loc

    def set_jitter(self, h_j, v_j):
        assert h_j.shape == v_j.shape, "Jitter ndim is diffient"
        self.fov_x_jitter = h_j
        self.fov_y_jitter = v_j

    def stitch(self, src_fovs: dict, output_path: str = None, stitch=True):
        '''
        src_fovs: {'row_col':'image_path'}
        output_path:
        stitch: 是否拼接图像
        '''

        self._init_parm(src_fovs)

        # 求解偏移矩阵
        jitter_model = FOVAligner(src_fovs, self.rows, self.cols, multi=True)
        if self.fov_x_jitter is None or self.fov_y_jitter is None:
            start_time = time.time()
            glog.info('Start jitter mode.')
            jitter_model.create_jitter()
            self.fov_x_jitter = jitter_model.horizontal_jitter
            self.fov_y_jitter = jitter_model.vertical_jitter

            self.__jitter_diff = jitter_model._offset_eval(self._fov_height, self._fov_width, self._overlap)
            end_time = time.time()
            glog.info("Caculate jitter time -- {}s".format(end_time - start_time))
        else:
            self.__jitter_diff = jitter_model._offset_eval(self._fov_height, self._fov_width, self._overlap)
            glog.info("Have jitter matrixs, skip this mode.")

        # 求解拼接坐标
        if self.fov_location is None:
            start_time = time.time()
            glog.info('Start location mode.')
            location_model = GlobalLocation()
            location_model.set_size(self.rows, self.cols)
            location_model.set_image_shape(self._fov_height, self._fov_width)
            location_model.set_jitter(self.fov_x_jitter, self.fov_y_jitter)
            location_model.create_location()
            self.fov_location = location_model.fov_loc_array
            self.__offset_diff = location_model.offset_diff
            end_time = time.time()
            glog.info("Caculate location time -- {}s".format(end_time - start_time))
        else:
            glog.info("Have location coord, skip this mode.")

        # 拼接
        if stitch and output_path is not None:
            start_time = time.time()
            glog.info('Start stitch mode.')
            wsi = StitchingWSI()
            wsi.set_overlap(self._overlap)
            wsi.mosaic(src_fovs, self.fov_location, multi=False)
            wsi.save(output_path, compression=False)
            end_time = time.time()
            glog.info("Stitch image time -- {}s".format(end_time - start_time))

            self.__image = wsi.buffer

    def get_template(self, ): return self._template

    def template(self, pts, scale_x, scale_y, rotate, chipno, index, output_path=None):

        tr = TemplateReference()
        tr.set_scale(scale_x, scale_y)
        tr.set_rotate(rotate)
        tr.set_chipno(chipno)
        tr.set_fov_location(self.fov_location)
        tr.set_qc_points(index, pts)

        tr.reference_template(mode='multi')
        self._template = tr.template
        if output_path is not None: tr.save_template(output_path)
        max_value, mean_value = tr.get_template_eval()
        self.__template_max_value = max_value
        self.__template_mean_value = mean_value
        glog.info("Reference template -- max value:{}  mean value:{}".format(max_value, mean_value))
        return self._template, [tr.scale_x, tr.scale_y, tr.rotation]

    def get_image(self):
        return self.__image

    def get_stitch_eval(self):
        eval = dict()

        eval['stitch_diff'] = self.__offset_diff
        eval['jitter_diff'] = self.__jitter_diff

        eval['stitch_diff_max'] = max(self.__offset_diff)
        eval['jitter_diff_max'] = max(self.__jitter_diff)

        eval['template_max'] = self.__template_max_value
        eval['template_mean'] = self.__template_mean_value

        return eval

    def mosaic(self, ):
        pass
