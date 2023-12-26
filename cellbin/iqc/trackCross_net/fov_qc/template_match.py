import glob
from json import load, dump, dumps
from copy import copy
import math
import numpy as np
import cv2 as cv
from os.path import join
from ..common import filename2index
from .shape import Line
from .map import get_id_scale, map_by_intercept


class Template(object):
    def __init__(self, ):
        self.scale = 1.0
        self.rotation = 1.0
        self.chip_template = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
                              [112, 144, 208, 224, 224, 208, 144, 112, 160]]
        self.track_lines = None
        self.templx_lines = None
        self.temply_lines = None
        self.templ_points = None

    def set_chip_template(self, template):
        self.chip_template = template

    # def load_track_lines(self, label_file):
    #     self.track_lines = list()
    #     line = Line()
    #     with open(label_file, 'r') as fd:
    #         dct = load(fd)
    #         for shape in dct['shapes']:
    #             if shape['shape_type'] == 'line':
    #                 pt0, pt1 = shape['points']
    #                 line.init_by_point_pair(pt0, pt1)
    #                 self.track_lines.append(copy(line))

    def template_match(self, x_intercept, y_intercept, shape):
        def index_match(intercept, template):
            index = list()
            interval = [(intercept[i+1] - intercept[i]) for i in range(len(intercept) - 1)]
            count = len(interval)
            template_len = len(template)
            template_used = template * (math.ceil(count / template_len) + 1)
            std_var = list()
            for i in range(len(template)):
                items = template_used[i: i + count]
                std_var.append(np.std([(interval[i] / items[i]) for i in range(count)]))
            index0 = np.argmin(std_var)
            for i in range(len(intercept)): index.append((index0 + i) % template_len)

            template__len = 0
            for i in index[:-1]: template__len += template[i]
            len_ = np.max(intercept) - np.min(intercept)
            scale_ = len_ / template__len
            return index, scale_

        def template_reassign(index, template, scale):
            tr = []
            for i in range(len(index)):
                if index[i] == -1:
                    continue
                dist = template[index[i]]
                try:
                    dd = [k >= 0 for k in index[i+1:]].index(True)
                except:
                    dd = -1

                if dd >= 0:
                    j = index[i] + 1
                    if j >= len(template):
                        j = j % len(template)
                    while j != index[i + dd + 1]:
                        dist += template[j]
                        j += 1
                        if j >= len(template):
                            j = j % len(template)
                tr.append(dist * scale)

            # for i in range(len(index)):
            #     if index[i] == -1:
            #         continue
            #     elif i == 0:
            #         tr.append(template[index[i]] * scale)
            #         continue
            #     else:
            #         for _i_ in range(1, i + 1):
            #             if index[i - _i_] == -1:
            #                 if i == 1:
            #                     tr.append(template[index[i]] * scale)
            #                     break
            #                 else:
            #                     continue
            #             else:
            #                 tr.append(np.sum(template[index[i] - _i_ + 1:index[i] + 1]) * scale)
            #                 break

            return tr

        def intercept_reassign(x0, intercept_):
            intercept = [x0]
            accumulator = x0
            for i in intercept_:
                accumulator += i
                intercept.append(accumulator)
            return intercept

        x_template = self.chip_template[0]
        y_template = self.chip_template[1]
        # x_index_1, x_scale_1 = index_match(x_intercept, x_template)
        # y_index_1, y_scale_1 = index_match(y_intercept, y_template)

        if len(x_intercept) == 0 or len(y_intercept) == 0:
            return -1, -1, -1, -1
        # x_index, x_scale = get_id_scale(x_intercept, x_template)
        # y_index, y_scale = get_id_scale(y_intercept, y_template)
        x_index, x_scale = map_by_intercept(x_intercept, x_template, shape[1])
        y_index, y_scale = map_by_intercept(y_intercept, y_template, shape[0])

        if x_scale < 0 or y_scale < 0:
            return -1, -1, -1, -1

        if abs(x_scale / y_scale - 1) > 0.1 or len(x_index) < 2 or len(y_index) < 2:
            return -1, -1, -1, -1

        if len(y_index) < len(x_index):
            y_scale += (1 - len(y_index) / len(x_index)) * abs(x_scale - y_scale)
        elif len(y_index) > len(x_index):
            x_scale += (1 - len(x_index) / len(y_index)) * abs(x_scale - y_scale)

        self.scale = (x_scale + y_scale) / 2
        # self.scale = x_scale
        # x_i_1 = [(x_template[x] * self.scale) for x in x_index]
        # y_i_1 = [(y_template[y] * self.scale) for y in y_index]

        x_i = template_reassign(x_index, x_template, x_scale)
        y_i = template_reassign(y_index, y_template, y_scale)

        x_intercept = [x_intercept[x_] for x_ in range(len(x_intercept)) if x_index[x_] >= 0]
        y_intercept = [y_intercept[y_] for y_ in range(len(y_intercept)) if y_index[y_] >= 0]

        x_intercept_ = intercept_reassign(x_intercept[0], x_i[:-1])
        y_intercept_ = intercept_reassign(y_intercept[0], y_i[:-1])

        return x_intercept_, y_intercept_, x_index, y_index

    def match(self, shape):
        self.shape = shape
        # self.load_track_lines(label_file)
        k = list()
        x_intercept = list()
        y_intercept = list()
        counter = 0
        for line in self.track_lines:
            if abs(line.coefficient) <= 1:
                k.append(line.coefficient)
                y_intercept.append([line.bias, counter])
                counter += 1
            else:
                k.append(-1 / line.coefficient)
                x_intercept.append([line.get_point_by_y(0)[0], counter])
                counter += 1
        x_intercept.sort()
        y_intercept.sort()

        x_intercept_, y_intercept_, x_index, y_index = self.template_match([val[0] for val in x_intercept], [val[0] for val in y_intercept], shape)
        if x_intercept_ == -1:
            return -1

        self.templx_lines = list()
        self.temply_lines = list()
        line = Line()

        strip = []
        for i in range(len(x_index)):
            if x_index[i] == -1:
                strip.append(x_intercept[i][1])
        for j in range(len(y_index)):
            if y_index[j] == -1:
                strip.append(y_intercept[j][1])

        k = [k[i] for i in range(len(k)) if i not in strip]

        coeff = np.mean(k)
        self.rotation = math.degrees(math.atan(coeff))
        self.scale = self.scale * math.cos(math.radians(self.rotation))

        x_index = [x for x in x_index if x >= 0]
        y_index = [y for y in y_index if y >= 0]

        for i in range(len(x_intercept_)):
            x = x_intercept_[i]
            line.init_by_point_k([x, 0], -1 / coeff)
            line.index = x_index[i]
            self.templx_lines.append(copy(line))
        for j in range(len(y_intercept_)):
            y = y_intercept_[j]
            line.init_by_point_k([0, y], coeff)
            line.index = y_index[j]
            self.temply_lines.append(copy(line))
        self.make_cross_points()
        self.point_spread_into_template()
        return 0

    def point_spread_into_template(self):
        x, y, ind_x, ind_y = self.templ_points[0]
        # print(x, y, ind_x, ind_y)
        if (self.rotation >= 0) and (self.rotation < 45):
            r = self.rotation - 90
        elif (self.rotation < 0) and (self.rotation > -45):
            r = self.rotation + 90
        k0 = math.tan(math.radians(self.rotation))
        if k0 == 0: k0 = 0.00000001
        k1 = -1 / k0
        x0, y0 = 0, 0
        x0 += x
        y0 += y

        y_intercept0 = y0 - k0 * x0
        x_intercept0 = (y0 - k1 * x0) * k0

        dy = abs(k0 * self.shape[1])
        y_region = (-dy, self.shape[0] + dy)
        dx = abs(k0 * self.shape[0])
        x_region = (-dx, self.shape[1] + dx)
        self.y_intercept = self.get_intercept(y_intercept0, y_region, ind_y, self.chip_template[1])
        self.x_intercept = self.get_intercept(x_intercept0, x_region, ind_x, self.chip_template[1])
        self.create_cross_points(k0)

    def get_intercept(self, intercept0, region, ind, templ):
        self.item_count = len(self.chip_template[0])
        idx = intercept0
        intercept = [[idx, ind]]
        s, e = region
        # face to large
        while idx < e:
            ind = ind % self.item_count
            item_len = (templ[ind] * self.scale) / math.cos(math.radians(self.rotation))
            idx += item_len
            intercept.append([idx, (ind + 1) % self.item_count])
            ind += 1
        # face to small
        idx, ind = intercept[0]
        while idx > s:
            ind -= 1
            ind = ind % self.item_count
            item_len = (templ[ind] * self.scale) / math.cos(math.radians(self.rotation))
            idx -= item_len
            intercept.append([idx, ind])
        return sorted(intercept, key=(lambda x: x[0]))

    def create_cross_points(self, k):
        self.cross_points = list()
        for x_ in self.x_intercept:
            for y_ in self.y_intercept:
                x, ind_x = x_
                y, ind_y = y_
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                if x0 < 0 or x0 > self.shape[1] or y0 < 0 or y0 > self.shape[0]:
                    continue
                self.cross_points.append([x0, y0, ind_x, ind_y])

    def make_cross_points(self, ):
        self.templ_points = list()
        for yl in self.temply_lines:
            for xl in self.templx_lines:
                index_y = yl.index
                index_x = xl.index
                y = yl.bias
                k = yl.coefficient
                x = xl.get_point_by_y(0)[0]
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                self.templ_points.append([x0, y0, index_x, index_y])

    def display(self, arr):
            h, w, c = arr.shape
            # for line in self.track_lines:
            #     p0, p1 = line.two_points((h, w))
            #     cv.line(arr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 255, 0), 1)
            for line in self.templx_lines:
                p0, p1 = line.two_points((h, w))
                cv.line(arr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 255), 1)
            for line in self.temply_lines:
                p0, p1 = line.two_points((h, w))
                cv.line(arr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 255), 1)

            for p in self.templ_points:
                x, y, x_index, y_index = p
                cv.circle(arr, (int(x), int(y)), 16, (255, 0, 255), 2)
            cv.imwrite('demo.tif', arr)

    def templ_cross_str(self, r, c):
        lines = '{}_{}\t1.0\n'.format(str(r).zfill(4), str(c).zfill(4))
        for p in self.templ_points:
            x, y, x_index, y_index = p
            lines += '{}\t{}\t{}\t{}\n'.format(x, y, x_index, y_index)
        return lines


def make_templ_cross_points(json_path, style, chip=None):
    tpl = Template()
    if chip is not None: tpl.chip_template = chip
    labels = glob.glob(join(json_path, '*.json'))
    points_str = ''
    rotation = 0
    scale = 0
    row = 0
    col = 0
    for label in labels:
        c, r = filename2index(label, style)
        flag = False
        with open(label, 'r') as fd:
            shapes = load(fd)['shapes']
            if len(shapes) > 0:
                fov_str = '{}_{}\t0.0\n'.format(str(r).zfill(4), str(c).zfill(4))
                for s in shapes:
                    if s['shape_type'] == 'line': flag = True
                    x, y = s['points'][0]
                    fov_str += '{}\t{}\t{}\t{}\n'.format(x, y, 0, 0)
            if flag:
                tpl.manual_from_file(label)
                points_str += tpl.templ_cross_str(r, c)
                rotation = tpl.rotation
                scale = tpl.scale
                row = r
                col = c
            else:
                if len(shapes): points_str += fov_str
    return points_str, rotation, scale, row, col


def main():
    json_path = 'D:\DATA\stitchingv2_test\motic\mouse_heart_eu\DP8400016861TR_C3\DP8400016861TR_C3_Neo1.imgs'
    tpl = Template()
    tpl.manual_from_file(json_path+'\\0005_0004.json')
    arr = cv.imread(json_path+'\\0005_0004.png', -1)
    tpl.display(arr)
    # tpl.templ_cross_str(0, 4)
    points_str, rotation, scale, row, col = make_templ_cross_points(json_path, 'motic')
    print(points_str)


if __name__ == '__main__':
    main()
