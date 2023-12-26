###################################################
"""reference template for image, must need QC data.
create by lizepeng, 2023/1/31 10:00
"""
####################################################

import copy

import glog
import numpy as np
import os
import math


class TemplateReference:
    """ Template Reference """
    def __init__(self, ):
        """Initialize Template Reference.

        Args:
          chip_no: STOmics chip number
          scale_x: The X-direction scale factor of the stain image relative to the gene matrix.
          scale_y: The Y-direction scale factor of the stain image relative to the gene matrix.
          rotation: The rotation factor of the stain image relative to the gene matrix.
          qc_pts: Anchor points detect by imageQC module.
          template_qc_pts:
          x_intercept: Gene matrix line spacing in the horizontal direction.
          y_intercept: Gene matrix line spacing in the vertical direction.
          fov_loc_array: Each FOV space coordinate matrix.
          template_center_pt: The starting local space point for template construction.
          template_qc:
          mosaic_height: Height of mosaic image.
          mosaic_width: Width of mosaic image.
          template: Total anchor points of mosaic image
          template_src:
        """
        self.chip_no: list = list()
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.rotation: float = 0.0
        self.qc_pts: dict = {}
        self.template_qc_pts: dict = {}

        #
        self.x_intercept = None
        self.y_intercept = None
        self.fov_loc_array = None
        self.template_center_pt: list = []
        self.template_qc: list = []
        self.mosaic_height = None
        self.mosaic_width = None
        self.__min_distance = None  # The current minimum distance value in the minimum distance method

        # output
        self.template: list = list()
        self.template_src = None

    ################
    '''init parm'''
    ################
    def set_scale(self, scale_x: float, scale_y: float):
        self.scale_x = self.__to_digit(scale_x)
        self.scale_y = self.__to_digit(scale_y)
        assert self.scale_x is not None and self.scale_y is not None, "Input is not a number."

    def set_rotate(self, r: float):
        self.rotation = self.__to_digit(r)
        assert self.rotation is not None, "Input is not a number."

    def set_chipno(self, chip_no):
        '''
        :param chip_no: Chip standard cycle
        :return:
        '''
        assert type(chip_no) == list or type(chip_no) == np.ndarray, "ChipNO must be a list or array."
        self.chip_no = chip_no

    def set_fov_location(self, global_loc):
        '''
        :param global_loc: global stitching coordinates
        :return:
        '''
        self.fov_loc_array = global_loc
        self.mosaic_width, self.mosaic_height = self.fov_loc_array[-1, -1] + self.fov_loc_array[1, 1]

    def set_qc_points(self, index, pts):
        """
        index: template FOV , row_col or [row, col]
        pts: {index: [x, y, ind_x, ind_y], ...}
        """
        if self.fov_loc_array is None:
            print("Please init global location.")
            return

        if isinstance(index, str):
            row, col = index.split("_")
        elif isinstance(index, list):
            row, col = index
        else:
            print("FOV index error.")
            return

        row = int(row)
        col = int(col)

        assert isinstance(pts, dict) or isinstance(pts, h5py.Group), "QC Points is error."
        for ind in pts.keys():
            points = np.array(pts[ind])
            if len(points) > 0:
                if ind != index:
                    self.qc_pts[ind] = points
                else:
                    self.template_qc_pts[ind] = points

        self.template_center_pt = copy.deepcopy(self.template_qc_pts[index][0])
        # self.template_center_pt = np.array([2206, 1665, 0, 0])
        self.template_center_pt[:2] += self.fov_loc_array[row, col]

    ################
    '''reference'''
    ################
    def __delete_outline_points(self, points_re, points_qc, range_size=5000):
        '''
        Outlier removal
        :param points_re:
        :param points_qc:
        :param range_size: Marquee size
        :return:
        '''
        _points_qc = list()
        _points_re = list()
        for k, point in enumerate(points_qc):
            if np.abs(point[0] - self.template_center_pt[0]) <= range_size and \
                    np.abs(point[1] - self.template_center_pt[1]) <= range_size:
                _points_qc.append(point)
                _points_re.append(points_re[k])

        return np.array(_points_re), np.array(_points_qc)

    def __check_parm(self):
        assert self.scale_x is not None and self.scale_y is not None, "Scale is need init."
        assert self.rotation is not None, "Rotate is need init."
        assert self.chip_no is not None and len(self.chip_no) != 0, "ChipNO is need init."
        # assert len(self.qc_pts) != 0 and len(self.template_qc_pts) != 0, "QC points is need init."

    def __global_qc_points_to_global(self):
        """ Use when the QC point is a global point Generally not used """
        points_list = [self.qc_pts]
        for type_points in points_list:
            for fov_name in type_points.keys():
                row, col = [int(i) for i in fov_name.split("_")]
                for point in type_points[fov_name]:
                    temp = copy.deepcopy(point)
                    self.template_qc.append(temp[:2])
                    break

    def __qc_points_to_gloabal(self):
        """ QC points mapped to global coordinates """
        points_list = [self.qc_pts]
        for type_points in points_list:
            for fov_name in type_points.keys():
                row, col = [int(i) for i in fov_name.split("_")]
                if type_points[fov_name].ndim == 1:
                    temp = copy.deepcopy(type_points[fov_name])
                    temp[:2] += self.fov_loc_array[row, col]
                    self.template_qc.append(temp[:2])
                else:
                    for point in type_points[fov_name]:
                        temp = copy.deepcopy(point)
                        temp[:2] += self.fov_loc_array[row, col]
                        self.template_qc.append(temp[:2])
                        break
    @staticmethod
    def pair_to_template(temp_qc, temp_re, threshold=50): # TODO 临时变量
        '''one point of temp0 map to only one point of temp1'''
        import scipy.spatial as spt

        temp_src = np.array(temp_re)[:, :2]
        temp_dst = np.array(temp_qc)[:, :2]
        tree = spt.cKDTree(data=temp_src)
        distance, index = tree.query(temp_dst, k=1)

        thr_index = index[distance < threshold]
        points_qc = temp_dst[distance < threshold]
        points_re = np.array(temp_re)[thr_index]

        return [points_re, points_qc]

    @staticmethod
    def resove_affine_matrix(H):
        theta = (math.degrees(math.atan2(H[1, 0], H[0, 0])) + math.degrees(math.atan2(H[1, 1], H[0, 1])) - 90) / 2
        s_x = math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        s_y = (H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]) / s_x
        confidence = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / s_x
        print("result: \ntheta: {}, \ns_x: {}, \ns_y: {}, \nconfidence:{}".format(theta, s_x, s_y, 1 - abs(confidence)))
        return theta, s_x, s_y

    def __mean_to_scale_and_rotate(self, points_re, points_qc):
        """ Find the mean value of the scale and rotate differences between the template point and the QC point """
        scale_x_list = []
        scale_y_list = []
        rotate_list = []
        for point_re, point_qc in zip(points_re, points_qc):

            if point_qc[0] != self.template_center_pt[0]:

                #旋转角
                rotation_dst = math.degrees(
                    math.atan((point_qc[1] - self.template_center_pt[1]) / (point_qc[0] - self.template_center_pt[0])))
                rotation_src = math.degrees(
                    math.atan((point_re[1] - self.template_center_pt[1]) / (point_re[0] - self.template_center_pt[0])))

                _rotate = self.rotation + (rotation_dst - rotation_src)

                src_x = point_re[0] - self.template_center_pt[0]
                src_y = point_re[1] - self.template_center_pt[1]

                dst_x = point_qc[0] - self.template_center_pt[0]
                dst_y = point_qc[1] - self.template_center_pt[1]

                dis = math.sqrt(src_x ** 2 + src_y ** 2)

                _src_x = dis * math.cos(math.radians(np.abs(rotation_dst)))
                _src_y = dis * math.sin(math.radians(np.abs(rotation_dst)))

                _scale_x = self.scale_x / np.abs(_src_x / dst_x)
                _scale_y = self.scale_y / np.abs(_src_y / dst_y)

                scale_x_list.append(_scale_x)
                scale_y_list.append(_scale_y)
                rotate_list.append(_rotate)

        return np.mean(scale_x_list), np.mean(scale_y_list), np.mean(rotate_list)

    @staticmethod
    def __leastsq_to_scale_and_rotate(point_re, point_qc):
        """ Minimize the distance between the template point and the QC point and solve the result """
        # point_re = np.array([[61.237, 35.355], [-35.355, 61.237], [-61.237, -35.355], [35.355, -61.237]])
        # point_qc = np.array([[100, 100], [-100, 100], [-100, -100], [100, -100]])
        from scipy.optimize import leastsq, minimize

        for k, point in enumerate(point_re):
            if point[0] == 0 and point[1] == 0:
                point_re = np.delete(point_re, k, axis=0)
                point_qc = np.delete(point_qc, k, axis=0)
                break

        def _error(p, point_re, point_qc):
            _scale_x, _scale_y, _rotate = p

            src_x = point_re[:, 0]
            src_y = point_re[:, 1]

            _t = (src_y) / (src_x)
            _d = [math.atan(i) for i in _t]
            rotation_src = np.array([math.degrees(i) for i in _d])

            src_x = point_re[:, 0] * (1 + _scale_x)
            src_y = point_re[:, 1] * (1 + _scale_y)

            dis = [math.sqrt(i) for i in src_x ** 2 + src_y ** 2]

            dst_x = dis * np.array([math.cos(math.radians(np.abs(i))) for i in (rotation_src + _rotate)])
            dst_y = dis * np.array([math.sin(math.radians(np.abs(i))) for i in (rotation_src + _rotate)])

            dst_x = [-i if point_re[k, 0] < 0 else i for k, i in
                     enumerate(dst_x)]
            dst_y = [-i if point_re[k, 1] < 0 else i for k, i in
                     enumerate(dst_y)]

            error = (point_qc[:, 0] - dst_x) ** 2 + (point_qc[:, 1] - dst_y) ** 2
            error = [math.sqrt(i) for i in error]
            # print(np.sum(error))
            return np.sum(error)

        para = minimize(_error, x0=np.zeros(3, ), args=(point_re, point_qc))

        return para

    def __caculate_scale_and_rotate(self, points_re, points_qc, mode='minimize', update=True):
        """ Calculate scale and rotate using template points and QC points """
        if points_re.shape[1] == 4:
            points_re = points_re[:, :2]

        points_re[:, 0] -= self.template_center_pt[0]
        points_re[:, 1] = self.template_center_pt[1] - points_re[:, 1]

        points_qc[:, 0] -= self.template_center_pt[0]
        points_qc[:, 1] = self.template_center_pt[1] - points_qc[:, 1]

        if mode == 'minimize':
            # Minimum distance optimization method
            para = self.__leastsq_to_scale_and_rotate(points_re, points_qc)
            _scale_x, _scale_y, _rotate = para.x
            self.__min_distance = para.fun / len(points_re)
            if update:
                self.rotation -= _rotate
                self.scale_x *= (1 + _scale_x)
                self.scale_y *= (1 + _scale_y)
        elif mode == 'mean':
            # mean method
            scale_x, scale_y, rotate = self.__mean_to_scale_and_rotate(points_re, points_qc)
            if update:
                self.rotation = rotate
                self.scale_x = scale_x
                self.scale_y = scale_y
        else: pass

    def reference_template(self, mode='single'):
        '''
        mode: have three type ['single', 'double', 'multi']
        *   single: only reference template FOV
        *   double: reference template FOV & minimize the points distance
        *   multi: reference template FOV & minimize the points distance & change the template center point
        '''
        self.__check_parm()
        self.__point_inference(self.template_center_pt, (self.mosaic_height, self.mosaic_width))
        range_size = 5000  # TODO 取点范围
        max_item = int(max(self.mosaic_height, self.mosaic_width) / range_size)

        if mode != 'single':
            self.__qc_points_to_gloabal()
            # self.__global_qc_points_to_global()
            points_qc = np.zeros([0, 2])
            count = 0
            while count < max_item:
                points_re, points_qc = self.pair_to_template(self.template_qc, self.template)

                points_re, points_qc = self.__delete_outline_points(points_re, points_qc, range_size * (count + 1))
                self.__caculate_scale_and_rotate(points_re, points_qc)

                self.template = list()
                self.__point_inference(self.template_center_pt, (self.mosaic_height, self.mosaic_width))
                count += 1

            double_info = {'dis': self.__min_distance, 'point': self.template_center_pt,
                           'scale_x': self.scale_x, 'scale_y': self.scale_y,
                           'rotate': self.rotation}

            if mode == 'multi':
                current_min_distance = self.__min_distance  # 首次推导的最小距离值
                _points_qc = np.concatenate((points_qc, points_re[:, 2:]), axis=1)
                candidate_points = self.__find_center_close_point(_points_qc)
                candidate_info = list()

                for center_point in candidate_points:
                    self.__point_inference(center_point, (self.mosaic_height, self.mosaic_width))

                    # 循环推导
                    points_re, points_qc = self.pair_to_template(self.template_qc, self.template)
                    # points_re, points_qc = self.__delete_outline_points(points_re, points_qc)
                    self.template_center_pt = center_point
                    self.__caculate_scale_and_rotate(points_re, points_qc, update=True)
                    candidate_info.append({'dis':self.__min_distance, 'point': center_point,
                                           'scale_x':self.scale_x, 'scale_y':self.scale_y,
                                           'rotate':self.rotation})

                candidate_info = sorted(candidate_info, key=lambda x:x['dis'])
                min_info = candidate_info[0]

                if min_info['dis'] < current_min_distance:
                    result_info = min_info
                else: result_info = double_info

                self.scale_x = result_info['scale_x']
                self.scale_y = result_info['scale_y']
                self.rotation = result_info['rotate']
                self.__point_inference(result_info['point'], (self.mosaic_height, self.mosaic_width))

        glog.info("Reference template done")

    def get_template_eval(self):
        '''
        :return:  max(dis), mean(dis) Obtain the maximum value and average value of the template and QC points at this time
        '''
        points_re, points_qc = self.pair_to_template(self.template_qc, self.template)
        distances = list()
        for point_re, point_qc in zip(points_re, points_qc):
            dis = np.sqrt((point_re[0] - point_qc[0]) ** 2 + (point_re[1] - point_qc[1]) ** 2)
            distances.append(dis)

        return max(distances), np.mean(distances)

    def __find_center_close_point(self, points, n=5):
        '''
        Find the derived template point closest to the center point of the full graph
        :param points: [x, y, ind_x, ind_y]
        :return: self.template_center_pt
        '''
        points[:, 0] = self.template_center_pt[0] + points[:, 0]
        points[:, 1] = self.template_center_pt[1] - points[:, 1]

        center_point_x = self.mosaic_width / 2
        center_point_y = self.mosaic_height / 2

        dis_list = list()
        for point in points:
            dis = np.sqrt((point[0] - center_point_x) ** 2 + (point[1] - center_point_y) ** 2)
            dis_list.append(dis)

        min_points = np.array(dis_list).argsort()[:n]
        # min_points = np.array(dis_list).argsort()[-n:][::-1]

        return points[min_points]

    def __point_inference(self, src_pt: tuple, region: tuple):
        '''
        search stand template from bin file by key(chip_no).
        src_pt :(x, y, ind_x, ind_y)
        region: (height, width)
        '''
        if len(self.template) > 0:
            self.template = list()

        x0, y0, ind_x, ind_y = src_pt

        k0 = np.tan(np.radians(self.rotation))
        if k0 == 0: k0 = 0.00000001
        k1 = -1 / k0

        y_intercept0 = y0 - k0 * x0
        x_intercept0 = (y0 - k1 * x0) * k0

        dy = abs(k0 * region[1])
        y_region = (-dy, region[0] + dy)
        dx = abs(k0 * region[0])
        x_region = (-dx, region[1] + dx)

        self.y_intercept = self.__get_intercept(self.scale_y, y_intercept0, y_region, ind_y, self.chip_no[1])
        self.x_intercept = self.__get_intercept(self.scale_x, x_intercept0, x_region, ind_x, self.chip_no[0])
        self.__create_cross_points(k0)

    def __get_intercept(self, scale, intercept0, region, ind, templ):
        idx = intercept0
        intercept = [[idx, ind]]
        s, e = region
        item_count = len(templ)
        # face to large
        while idx < e:
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self.rotation))
            idx += item_len
            intercept.append([idx, ind + 1])
            ind += 1
        # face to small
        idx, ind = intercept[0]
        while idx > s:
            ind -= 1
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self.rotation))
            idx -= item_len
            intercept.append([idx, ind])
        return sorted(intercept, key=(lambda x: x[0]))

    def __create_cross_points(self, k):
        for x_ in self.x_intercept:
            for y_ in self.y_intercept:
                x, ind_x = x_
                y, ind_y = y_
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                self.template.append([x0, y0, ind_x, ind_y])

    def __to_digit(self, n):
        try: return float(n)
        except: return None

    ################
    '''output'''
    ################
    def save_template(self, output_path, template=None):
        '''
        :param output_path:
        :param template: Other templates can be passed in for saving
        :return: save template points
        '''
        if template is None:
            _template = self.template
        else: _template = template

        if _template is not None and len(_template) > 0:
            if not os.path.exists(output_path): os.makedirs(output_path)
            np.savetxt(os.path.join(output_path, 'template.txt'), np.array(_template))
        else:
            print("Template save failed.")


if __name__ == '__main__':
    import h5py

    ipr_path = r"D:\self_use\ImageStudio\git\new\CellBinFiles\SS200000737BL_B1D5_20221220_173216.ipr"
    pts = {}
    with h5py.File(ipr_path, "r+") as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        scalex = conf['Register'].attrs['ScaleX']
        scaley = conf['Register'].attrs['ScaleY']
        rotate = conf['Register'].attrs['Rotation']
        loc = conf['Stitch/BGIStitch/StitchedGlobalLoc'][...]
        # loc = conf['Stitch/ScopeStitch/GlobalLoc'][...]

    chipno = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
              [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    tr = TemplateReference()
    tr.set_scale(scalex, scaley)
    tr.set_rotate(rotate)
    tr.set_chipno(chipno)
    tr.set_fov_location(loc)
    tr.set_qc_points('0027_0004', pts)

    tr.reference_template(mode='multi')
    tr.save_template(r'')

    mat, template = tr.homography_image(r'')