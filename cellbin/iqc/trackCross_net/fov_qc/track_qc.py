import time
import cv2 as cv
import glog
import os
import sys
from math import floor, ceil
import numpy as np
from copy import deepcopy
import json
import math
import os.path as osp
import torch
from tqdm import tqdm

from .cross_detector import CrossPointDetector, PreProcess
from .track_line import TrackLine
from .template_match import Template
from .model_config import DetectorModelConfig, ClassificationModelConfig, LineDetectorModelConfig
from ..sold2.export_line_features import LineDetector
# from ImageQC.constant import PROCESS_NUM
from multiprocessing import process
"Global Variables"
SEED = 0
torch.manual_seed(SEED)


class QualityControl(object):
    def __init__(self, info, si):
        self.template_cross_final = dict()
        self.track_cross_all = dict()
        self.fov_info = info
        self.si = si
        self.image_angle = None
        self.rotation = None
        self.scale = None
        self.detector = None
        self.classify = None
        self.line_detector = None
        self.template_cross = None
        self.track_cross = None
        self.debug_save_path= None
        self.preprocess = None
        self.eval_path = None
        self.line_detect_debug = None
        self.track_line_debug = None
        self.template_debug_save = None
        self.cross_points_debug_save = None
        self.detector_config = DetectorModelConfig()
        self.line_detector_config = LineDetectorModelConfig()
        self.load_detector_model()
        self.load_line_detector_model()
        if self.si.QCInfo.Debug:
            self.debug_init()

    def debug_init(self):
        self.debug_save_path = osp.join(self.fov_info.scope_info.QCInfo.OutputPath,
                                        f"{self.fov_info.scope_info.QCInfo.ChipName}_qc_debug")
        os.makedirs(self.debug_save_path, exist_ok=True)
        glog.info(f"saving debug results -> {self.debug_save_path}")
        self.preprocess = PreProcess()
        self.eval_path = os.path.join(self.debug_save_path, "template_eval")
        os.makedirs(self.eval_path, exist_ok=True)
        self.line_detect_debug = os.path.join(self.debug_save_path, "line_detect")
        os.makedirs(self.line_detect_debug, exist_ok=True)
        self.track_line_debug = os.path.join(self.debug_save_path, "track_line")
        os.makedirs(self.track_line_debug, exist_ok=True)
        self.template_debug_save = os.path.join(self.debug_save_path, "template")
        os.makedirs(self.template_debug_save, exist_ok=True)
        self.cross_points_debug_save = os.path.join(self.debug_save_path, "final_result")
        os.makedirs(self.cross_points_debug_save, exist_ok=True)

    def load_detector_model(self):
        """load track cross detector model"""
        ratio = round(self.fov_info.fov_width / self.fov_info.fov_height, 1)
        net_height = 1024
        net_width = int(np.ceil(1024 / 256 * ratio) * 256)
        self.detector = CrossPointDetector(net_width, net_height,
                                           self.detector_config.confidence_score,
                                           self.detector_config.mns_threshold)
        self.detector.load_model(self.detector_config.net_path,
                                 self.detector_config.model_path,
                                 GPU=self.detector_config.GPU)
        glog.info('Load model successfully. [net_width, net_height, GPU] ({}, {}, {})'.format(net_width, net_height,
                                                                                              self.detector_config.GPU))

    def load_line_detector_model(self):
        """load line detection model"""
        self.line_detector = LineDetector(config_path=self.line_detector_config.config_path,
                                          multiscale=self.line_detector_config.multiscale,
                                          ckpt_path=self.line_detector_config.ckpt_path,)
        self.line_detector.load_model()

    def generate_template(self, cross_points, arr, img_name):
        arr = np.copy(arr)
        save_path = os.path.join(self.template_debug_save, img_name)
        arr = self.preprocess.process(arr, multichannel=True)
        for cps in cross_points:
            x, y, x_ind, y_ind = int(cps[0]), int(cps[1]), cps[2], cps[3]
            cv.drawMarker(arr, (x, y), (20, 1, 170), cv.MARKER_CROSS, 50, thickness=2)
        cv.imwrite(save_path, arr)

    def get_line_detect_data(self, cross_points, arr, img_name):
        data = []
        for cross_point in cross_points:
            (x, y), score = cross_point
            y, x = x, y
            r = 32
            x_tl = x - r
            y_tl = y - r
            x_rb = x + r
            y_rb = y + r
            if x_rb > arr.shape[0] or x_tl < 0 or y_rb > arr.shape[1] or y_tl < 0:
                continue
            cut_img = arr[x_tl: x_rb, y_tl: y_rb]
            cur_img_name = f"{img_name}@_{x_tl}_{x_rb}_{y_tl}_{y_rb}.tif"
            data.append([cut_img, cur_img_name])
        return data

    def line_detect(self, t):
        line_detect_data = []
        top_images_ = []
        for i, v in enumerate(t):
            if i == 2:
                break
            r, c = [int(i) for i in v[0].split('_')]
            cur_image = self.fov_info.get_image(r, c)
            cur_image_name = osp.basename(str(self.fov_info.fov_spatial_src[int(r), int(c)], encoding='utf-8'))
            data = self.get_line_detect_data(v[1], cur_image, cur_image_name)
            line_detect_data.extend(data)
            top_images_.append([r, c])
        self.image_angle = self.line_detector(images_list=line_detect_data, output_path=self.line_detect_debug)
        self.si.QCInfo.PredAngle = self.image_angle
        self.si.QCInfo.LineDetect = top_images_
        glog.info(f"Pred angle -> {self.image_angle}")
        glog.info(f"Line detect using FOVs -> {top_images_}")

    def qc(self):
        glog.info(f"Detect cross points")
        t1 = time.time()
        dct = self.detect_cross_points()
        t2 = time.time()
        self.si.QCInfo.TimeCost.CrossPoint = int(t2 - t1)
        "将检测track点多的fov排在前面，优先进行track线检测"
        t = sorted(dct.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
        glog.info(f"Detect line")
        line_detect_start = time.time()
        self.line_detect(t)
        "找到图片旋转角"
        line_detect_end = time.time()
        glog.info(f"Detect line time cost -> {line_detect_end - line_detect_start}")
        self.si.QCInfo.TimeCost.LineDetect = int(line_detect_end - line_detect_start)
        self.template_cross = dict()
        tl = TrackLine()
        tm = Template()
        "通过芯片号找到对应的模板"
        tm.set_chip_template(self.fov_info.scope_info.ChipInfo.FOVTrackTemplate)
        # 一个FOV检测到三个点才算是检测到点的FOV
        good_fov = len([l for l in t if len(l[1]) > 2])
        d_count = min(len([l for l in t if len(l[1]) > 2]), 50)
        passed = d_count
        scale = []
        rotation = []
        glog.info(f"Traditional line detect")
        t3 = time.time()
        for i in tqdm(range(d_count)):
            k, v = t[i]
            r, c = [int(i) for i in k.split('_')]
            arr = self.fov_info.get_image(r, c)
            cur_image_name = osp.basename(str(self.fov_info.fov_spatial_src[int(r), int(c)], encoding='utf-8'))
            "track线检测"
            track_line_error = tl.generate(arr, self.image_angle, cur_image_name, output_path=self.track_line_debug)
            if track_line_error == -1:
                passed -= 1
                continue
            "模板匹配"
            tm.track_lines = tl.track_lines
            ret = tm.match(arr.shape)
            if ret < 0:
                passed -= 1
                continue
            scale.append(tm.scale)
            rotation.append(tm.rotation)
            self.template_cross[k] = tm.cross_points
            if self.si.QCInfo.Debug:
                self.generate_template(tm.cross_points, arr, cur_image_name)
        t4 = time.time()
        self.si.QCInfo.TimeCost.TraditionalLineDetect = int(t4 - t3)
        glog.info(f"match template fov: {passed}")
        "对scale和rotation进行找中位数以及平均值计算"
        if len(scale) > 0 and len(rotation) > 0:
            scale.sort()
            rotation.sort()
            midMin = floor((len(scale) - 1) * 0.4)
            midMax = ceil((len(scale) - 1) * 0.6)
            if midMin != midMax:
                self.scale = 1 / np.mean(scale[midMin:midMax + 1])
                self.rotation = np.mean(rotation[midMin:midMax + 1])
            else:
                self.scale = 1 / scale[midMin]
                self.rotation = rotation[midMin]
        else:
            self.scale = -1
            self.rotation = -1000
        glog.info(f"the rotation -> {self.rotation}, the scale -> {self.scale}")
        "self.track_cross_all中保存所有的fov对应track点的位置信息"
        for i in range(len(t)):
            k, v = t[i]
            if len(v) == 0:
                continue
            self.track_cross_all[k] = v
        "当找到的模板不为0时候才会去选择最好的"
        glog.info(f"Template eval")
        if len(self.template_cross) != 0:
            self.get_best()
        t5 = time.time()
        self.si.QCInfo.TimeCost.TemplateEval = int(t5 - t4)
        """
        计算得分
        可找到一个得分不为0的模板FOV视为得分60
        yolo找到的点的FOV占比为82%时为80分
        yolo找到的点的FOV占比为100%时为100分
        yolo可找到点的FOV越多得分越高
        """

        score = 0 if len(self.template_cross_final) == 0 else 60
        if good_fov / self.fov_info.fov_count <= 0.82:
            score += round(good_fov * 2000 / 82 / self.fov_info.fov_count)
        else:
            last_fov_count = good_fov - 0.82 * self.fov_info.fov_count
            score += 20 + round(last_fov_count * 2000 / 18 / self.fov_info.fov_count)
        # score += round((len(self.track_cross_all) / self.fov_info.fov_count) * 0.49, 2)
        glog.info(f"yolo found track point on {len(self.track_cross_all)} fovs")
        tc_str = self.save_corss_points()
        self.si.QCInfo.Angle = round(self.rotation)
        return self.rotation, self.scale, score, tc_str, self.track_cross_all, self.template_cross_final

    def get_best(self):
        """对每一个template的FOV进行打分并且最后进行排序"""
        for key, value in tqdm(self.template_cross.items()):
            template_points = value
            yolo_points = self.track_cross_all[key]
            if self.si.QCInfo.Debug:
                r, c = [int(i) for i in key.split('_')]
                arr = self.fov_info.get_image(r, c)
            else:
                arr = None
                r, c = None, None
            pair, score, match = self.score(template_points, yolo_points, 5, arr, r, c)
            "val会被保存成[得分，对应的模板点]的形式"
            self.template_cross[key] = [score, template_points]
        "然后会跟得分进行排序"
        self.template_cross = {k: v for k, v in
                               sorted(self.template_cross.items(), key=lambda item: item[1][0], reverse=True)}
        "找到得分最高的对应的key"
        remove = list(self.template_cross.keys())[0]
        "如果得分不等于0，才会在template_cross_final中加入模板FOV"
        if self.template_cross[remove][0] != 0:
            "在最后的template模板中加入该模板的值"
            self.template_cross_final[remove] = self.template_cross[remove]
            "并且要在所有track点的dict中抹去这个key"
            del self.track_cross_all[remove]

    def score(self, template, yolo, tol, arr, r, c):
        """
            yolo预测的点与模板推的点的距离如果小于tol
            那么视他为匹配上了，将距离加到总距离中
            最后通过 总距离/最大可能距离 计算得分
            yellow -> 未匹配的最好预测点
            red -> 未匹配的模板点 (可找到对应yolo点的)
            green -> 匹配的模板点
            blue -> 未匹配的模板点 (不可找到对应yolo点的)
        """
        if arr is not None:
            arr = self.preprocess.process(arr, multichannel=True)
        total_dif = 0
        pair = []
        match = 0
        for row, col, row_ind, col_ind in template:
            min_dis = 100
            min_i = None
            cur_best = None
            for i, value in enumerate(yolo):
                pos, score = value
                row_, col_ = pos
                row_dif = row - row_
                col_dif = col - col_
                cur_dis = math.sqrt(row_dif ** 2 + col_dif ** 2)
                if cur_dis < min_dis:
                    cur_best = int(row_), int(col_)
                if cur_dis < tol:
                    min_dis = cur_dis
                    min_i = i
            if min_i is not None:
                pair.append([[row, col], [yolo[min_i]], [min_dis]])
                total_dif += min_dis
                match += 1
                if arr is not None:
                    cv.circle(arr, (int(row), int(col)), 2, (0, 255, 0), 2)
            else:
                if arr is not None:
                    cv.circle(arr, (int(row), int(col)), 1, (0, 0, 255), 1)
                    if cur_best is not None:
                        cv.circle(arr, cur_best, 1, (0, 255, 255), 1)
                    else:
                        cv.circle(arr, (int(row), int(col)), 1, (255, 0, 0), 1)
        if match != 0:
            max_dif = tol * match
            score = 1 - (total_dif / max_dif)
        else:
            score = 0
        if r is not None and c is not None and arr is not None and self.eval_path is not None:
            cv.imwrite(os.path.join(self.eval_path, f"{r}_{c}.tif"), arr)
        return pair, score, match

    def save_corss_points(self):
        """
        用于将cross点转换成文本格式
        :param debug_mode: 开启后会保存track点结果到图片上，输出路径会有以芯片号命名的结尾是qc_debug的文件夹
        :return:返回track点文本格式
        """
        lines = ''
        for it in self.template_cross_final.items():
            k, v = it
            if self.si.QCInfo.Debug:
                r, c = k.split("_")
                r, c = int(r), int(c)
                arr = self.fov_info.get_image(r, c)
                img_name = osp.basename(str(self.fov_info.fov_spatial_src[int(r), int(c)], encoding='utf-8'))
                if arr is None:
                    continue
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                arr = self.preprocess.process(arr, multichannel=True)
            lines += '{}\t{}\n'.format(k, v[0])
            for p in v[1]:
                x, y, x_ind, y_ind = p
                if self.si.QCInfo.Debug:
                    # cv.drawMarker(arr, (int(float(x)), int(float(y))), (20, 1, 170), cv.MARKER_CROSS, 50,
                    #               thickness=2)
                    cv.circle(arr, (int(float(x)), int(float(y))), 1, (0, 0, 255), 1)
                    cv.circle(arr, (int(float(x)), int(float(y))), 10, (0, 0, 255), 1)
                lines += '{}\t{}\t{}\t{}\n'.format(x, y, x_ind, y_ind)
            if self.si.QCInfo.Debug:
                cv.imwrite(os.path.join(self.cross_points_debug_save, f"{img_name}"), arr)
        for it in self.track_cross_all.items():
            k, v = it
            if len(v) == 0:
                continue
            lines += '{}\t0.0\n'.format(k)
            if self.si.QCInfo.Debug:
                r, c = k.split("_")
                arr = self.fov_info.get_image(int(r), int(c))
                img_name = osp.basename(str(self.fov_info.fov_spatial_src[int(r), int(c)], encoding='utf-8'))
                if arr is None:
                    continue
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                arr = self.preprocess.process(arr, multichannel=True)
                count = 0
            for p in v:
                x, y = p[0]
                lines += '{}\t{}\t0\t0\n'.format(x, y)
                if self.si.QCInfo.Debug:
                #     if count != 0:
                #         continue
                    # cv.drawMarker(arr, (int(float(x)), int(float(y))), (0, 255, 0), cv.MARKER_CROSS, 50,
                    #               thickness=2)
                    cv.circle(arr, (int(float(x)), int(float(y))), 1, (0, 255, 0), 1)
                    cv.circle(arr, (int(float(x)), int(float(y))), 10, (0, 0, 255), 1)
                    count += 1
            if self.si.QCInfo.Debug:
                cv.imwrite(os.path.join(self.cross_points_debug_save, f"{img_name}"), arr)
        return lines

    def detect_cross_points(self, ):
        """
        对每个FOV进行track点预测
        :return: 返回dict，每个key代表fov的索引，对应的val是该fov检出的track点坐标
        """
        dct = dict()
        for i in tqdm(range(self.fov_info.fov_rows)):
            for j in range(self.fov_info.fov_cols):
                try:
                    arr = self.fov_info.get_image(i, j)
                    # fov_name = self.fov_info.fov_spatial_src[i, j].decode().split("\\")[-1].split(".")[0]
                    if arr is None:
                        continue
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    self.detector.inference(arr)
                    if len(self.detector.boxes) != 0:
                        # pred_from_list, score_dict = self.cross_points_classification(boxes=deepcopy(self.detector.boxes),
                        #                                                               image=deepcopy(self.detector.image),
                        #                                                               image_name=fov_name)
                        # adjust_cross_points = self.remove_bad_cross_points(pred_from_list, score_dict)
                        dct['{}_{}'.format(i, j)] = self.detector.cross_points
                        # glog.info(f"{i, j} after adjustment {len(adjust_cross_points)}")
                    else:
                        dct['{}_{}'.format(i, j)] = self.detector.cross_points
                except Exception as e:
                    print(i, j)
                    print(e)
                    pass
        return dct

