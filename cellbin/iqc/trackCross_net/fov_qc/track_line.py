import math
from math import radians, cos, sin
import copy
import cv2 as cv
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

model = LinearRegression()
import random
from .shape import Line
from .adjust_image import qc_entry
# from shape import Line
from json import load
from copy import deepcopy
import os


def random_color():
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    r = random.randint(0, 256)
    return (b, g, r)


class TrackLine(object):
    def __init__(self):
        self.arr = None
        self.grid = 100
        self.horizontal_candidate_pts = None
        self.vertical_candidate_pts = None
        self.horizontal_pts = None
        self.vertical_pts = None
        self.horizontal_color_pts = None
        self.vertical_color_pts = None
        self.v_lines = None
        self.h_lines = None
        self.track_lines = None
        self.h, self.w = None, None
        self.h_ori, self.w_ori = None, None
        self.angle = None
        self.image = None
        self.image_name = None
        self.output_path = None

    def generate(self, image, angle, image_name, output_path):
        self.output_path = output_path
        self.image_name = image_name
        self.image = image
        if angle is not None:
            self.angle = angle
            self.h_ori, self.w_ori = self.image.shape
            self.arr = qc_entry(image=self.image, angle=self.angle)
        else:
            self.arr = self.image
        self.h, self.w = self.arr.shape
        # 得到水平方向的点的坐标
        self.horizontal_candidate_pts = self.create_candidate_pts('x')
        # 得到垂直防线的点的坐标
        self.vertical_candidate_pts = self.create_candidate_pts('y')
        # 得到垂直角度
        v_angle = self.integer_angle(self.vertical_candidate_pts, 'y')
        if v_angle == -1000:
            return -1
        # 得到水平角度
        h_angle = self.integer_angle(self.horizontal_candidate_pts, 'x')
        if h_angle == -1000:
            return -1
        # 过滤
        self.horizontal_pts = self.select_pts_by_integer_angle(self.horizontal_candidate_pts, h_angle, tolerance=1)
        self.vertical_pts = self.select_pts_by_integer_angle(self.vertical_candidate_pts, v_angle, tolerance=1)
        # 分类
        if len(self.horizontal_pts) != 0:
            self.horizontal_color_pts = self.classify_points(self.horizontal_pts, h_angle, tolerance=1)
        if len(self.vertical_pts) != 0:
            self.vertical_color_pts = self.classify_points(self.vertical_pts, v_angle, tolerance=1)
        # 逆合成线
        if len(self.horizontal_color_pts) != 0:
            self.h_lines = self.points_to_line(self.horizontal_color_pts, tolerance=3)
        if len(self.vertical_color_pts) != 0:
            self.v_lines = self.points_to_line(self.vertical_color_pts, tolerance=3)
        if len(self.h_lines) == 0:
            self.track_lines = self.v_lines
        if len(self.v_lines) == 0:
            self.track_lines = self.h_lines
        self.track_lines = self.h_lines + self.v_lines
        if angle is not None and len(self.track_lines) != 0:
            self.adjust_track_line()

    def adjust_track_line(self, ):
        lines_to_be_adjusted = deepcopy(self.track_lines)
        self.track_lines = []
        for line in lines_to_be_adjusted:
            new_line = Line()
            p0, p1 = line.two_points((self.h, self.w))
            p0_new, p1_new = self.rotate(p0, -self.angle), self.rotate(p1, -self.angle)
            new_line.init_by_point_pair(p0_new, p1_new)
            self.track_lines.append(new_line)
        if self.output_path is not None:
            self.debug()

    def rotate(self, pt, angle):
        px, py = pt
        cx = int(self.w / 2)
        cy = int(self.h / 2)
        theta = angle
        rad = radians(theta)
        new_px = cx + float(px - cx) * cos(rad) + float(py - cy) * sin(rad)
        new_py = cy + -(float(px - cx) * sin(rad)) + float(py - cy) * cos(rad)
        x_offset, y_offset = (self.w_ori - self.w) / 2, (self.h_ori - self.h) / 2
        new_px += x_offset
        new_py += y_offset
        return int(new_px), int(new_py)

    def load(self, label_file):
        self.track_lines = list()
        line = Line()
        with open(label_file, 'r') as fd:
            dct = load(fd)
            for shape in dct['shapes']:
                if shape['shape_type'] == 'line':
                    pt0, pt1 = shape['points']
                    line.init_by_point_pair(pt0, pt1)
                    self.track_lines.append(copy.copy(line))

    @staticmethod
    def points_to_line(dct, tolerance=2):
        lines = list()
        for k, v in dct.items():
            # 少于两个的拟合不成直线
            if len(v) > tolerance:
                tmp = np.array(v)
                model.fit(tmp[:, 0].reshape(-1, 1), tmp[:, 1])
                line = Line()
                # 一个点，加上coef拟合直线
                # print(model.coef_[0])
                # print(tmp)
                line.init_by_point_k(v[0], model.coef_[0])
                lines.append(line)
        return lines

    def classify_points(self, candidate_pts, base_angle, tolerance=2):
        pts = copy.copy(candidate_pts)
        ind = 0
        dct = dict()
        while (len(pts) > 1):
            pts_, index = self.angle_line(base_angle, pts, tolerance)
            # 将找到的同一直线的点放入dct中
            dct[ind] = pts_
            # 根据angle_line中返回的index删除已找到的点
            pts = np.delete(np.array(pts), index, axis=0).tolist()
            ind += 1
        # 返回存有点的分类的dct
        return dct

    @staticmethod
    def angle_line(angle, points, tolerance=2):
        # 找跟points[0]在一条直线的点
        count = len(points)
        orignal_point = points[0]
        points_ = [points[0]]
        index = [0]
        for i in range(1, count):
            p = points[i]
            line = Line()
            line.init_by_point_pair(orignal_point, p)
            diff = abs(line.rotation() - angle)
            diff = (diff > 90) and (180 - diff) or diff
            if diff < tolerance:
                points_.append(p)
                index.append(i)
        # 返回这条线的所有点，以及对应的index号，index用于之后删除array中已找到的点
        return points_, index

    @staticmethod
    def select_pts_by_integer_angle(candidate_pts, base_angle, tolerance=2):
        # 过滤
        x_count = len(candidate_pts)
        # pts用来储存该方向的所有点
        pts = list()
        for i in range(0, x_count - 1):
            if len(candidate_pts[i]) > 100:
                continue
            # 遍历所有的采样区域
            pts_start = candidate_pts[i]
            pts_end = candidate_pts[i + 1]
            # 遍历pts_start中的所有的点
            for p0 in pts_start:
                # 遍历pts_end中所有点，计算p0与各点的距离
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                # 取绝对值
                d_ = np.abs(d)
                # 找到最短的距离的index
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                # 计算角度
                line.init_by_point_pair(p0, pts_end[ind[0]])
                # 如果角度小于tol，
                if abs(line.rotation() - base_angle) <= tolerance: pts.append(p0)
        return pts

    def debug(self, candidate_pts=False, pts=False, color_points=False, line=True):
        """
        candidate_pts: 所有的点
        pts： 过滤
        color_points: 分类
        line: 拟合成线
        """
        from .cross_detector import PreProcess
        p = PreProcess()
        image = p.process(self.image, multichannel=True)
        if candidate_pts:
            points = self.vertical_candidate_pts + self.horizontal_candidate_pts
            for pts_ in points:
                for pt in pts_:
                    pt = (int(pt[0]), int(pt[1]))
                    cv.circle(image, pt, 3, (255, 255, 0), -1)
        if pts:
            points = self.vertical_pts + self.horizontal_pts
            for pt in points:
                pt = (int(pt[0]), int(pt[1]))
                cv.circle(image, pt, 5, (255, 0, 0), -1)

        if color_points:
            for k, v in self.vertical_color_pts.items():
                color = random_color()
                for p in v:
                    pt = (int(p[0]), int(p[1]))
                    cv.circle(image, pt, 5, color, -1)
            for k, v in self.horizontal_color_pts.items():
                color = random_color()
                for p in v:
                    pt = (int(p[0]), int(p[1]))
                    cv.circle(image, pt, 5, color, -1)

        if line:
            for line_ in self.track_lines:
                p0, p1 = line_.two_points((self.h_ori, self.w_ori))
                cv.line(image, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 1)

        cv.imwrite(os.path.join(self.output_path, f"{self.image_name}.tif"), image)

    @staticmethod
    def integer_angle(pts, derection='x'):
        x_count = len(pts)
        # angles将保存该方向上所有采样区域的角度
        angles = list()
        for i in range(0, x_count - 1):
            if len(pts[i]) > 100:
                continue
            # 相邻的两个采样区域
            pts_start = pts[i]
            pts_end = pts[i + 1]
            for p0 in pts_start:
                # pts_start中的一个点p0对应pts_end的所有点的euclidean distance
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                # 取绝对值
                d_ = np.abs(d)
                # 找到pts_start的该点的最小距离的index
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                # 将p0和对应的最小距离的点输入到line.init_by_point_pair
                # 可以得到线的coef和bias
                line.init_by_point_pair(p0, pts_end[ind[0]])
                # rotation方法运用上面得到的coef计算该直线的角度
                # 并将角度记录
                angles.append(round(line.rotation()))
        if len(angles) == 0:
            return -1000
        x = np.array(angles) - np.min(angles)
        # u, c = np.unique(x, return_counts=True)
        # print(np.asarray((u, c)).T)
        # 得到该方向上的角度（出现最多的角度）
        return np.argmax(np.bincount(x)) + np.min(angles)

    def create_candidate_pts(self, derection='x'):
        mat = self.arr
        pts = list()
        h, w = mat.shape
        # direction x -> h
        # direction y -> w
        counter = (derection == 'x' and h or w)
        # self.grid -> defined by user, 采样间隔
        for i in range(0, counter, self.grid):
            # i -> 从0到h或者w
            # 设置t为当前采样间隔的x或者y
            t = i + self.grid / 2
            if derection == 'x':
                # 区域 -> i到i+采样间隔的区域
                region_mat = mat[i: i + self.grid, :w]
                # 如果区域不是取样规定长度，继续
                if region_mat.shape[0] != self.grid:
                    continue
                # 对y方向进行像素求和，并除以规定的采样间隔，可以看做成normalization
                line = np.sum(region_mat, axis=0) / self.grid
            else:
                # 这里的处理与上述基本一样，只是这里是方向y的情况
                region_mat = mat[:h, i: i + self.grid]
                if region_mat.shape[1] != self.grid:
                    continue
                line = np.sum(region_mat, axis=1) / self.grid
            # 找到该条线上的极值（最小值）
            p = argrelextrema(line, np.less_equal, order=100)
            # print(p[0].shape)
            if derection == 'x':
                pt = [[p, t] for p in p[0]]
            else:
                pt = [[t, p] for p in p[0]]
            # pts中保存的为该方向的所有点

            pts.append(pt)
        return pts


def main():
    image_path = r"D:\dzh\data\FP200003814_M9\FP200003814_M9\FP200003814_M9_0002_0003_2021-12-16_12-47-52-834.tif"
    arr = cv.imread(image_path, 0)
    # arr = cv.medianBlur(arr, 3)
    print(arr.shape)
    ftl = TrackLine()
    ftl.generate(arr)
    ftl.debug()


if __name__ == '__main__':
    main()
