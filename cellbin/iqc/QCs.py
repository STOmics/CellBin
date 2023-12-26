import time
import cv2 as cv
import os, sys
from os.path import dirname, abspath
import numpy as np
import math
import glob
import tifffile as tifi
from shutil import rmtree
from os.path import join, dirname
from settings import grids_x, Image_magnify, norm_magnify, ratio, Template
from trackCross_net.fov_info import FOVInfo
import trackCross_net.fov_qc.track_qc as fq
from ImageQualityAssessment import ImageQualityAssessment


def single_row_col(file):
    sets = os.path.splitext(os.path.basename(file))[0].split("_")
    if file.endswith('.txt'):
        try:
            row, col = int(float(sets[-5])), int(float(sets[-4]))
        except:
            row, col = int(float(sets[-3])), int(float(sets[-2]))
    else:
        try:
            row, col = int(float(sets[-2])), int(float(sets[-1]))
        except:
            try:
                row, col = int(float(sets[-4])), int(float(sets[-3]))
            except:
                try:
                    row, col = int(float(sets[-3])), int(float(sets[-2]))
                except:
                    if sets[0][0] == '0':
                        sets.insert(0, 'image')
                    for set in sets[1:]:
                        if set[0] != '0':
                            sets.remove(set)
                        if set[0] == '0':
                            break
                    try:
                        row, col = int(float(sets[1])), int(float(sets[2]))
                    except Exception as e:
                        print(e)
                        row, col = int(float(sets[2])), int(float(sets[3]))
    return row, col


def get_row_col(imgs_path):
    imgs_lst = glob.glob(os.path.join(imgs_path, '*.tif'))
    if not imgs_lst:
        imgs_lst = glob.glob(os.path.join(imgs_path, '*.png'))
    max_row, max_col = 0, 0
    for img_file in imgs_lst:
        row, col = single_row_col(img_file)
        max_row = row if row > max_row else max_row
        max_col = col if col > max_col else max_col

    return max_row, max_col


def cal_bad_num(PATH, si):
    # image processing
    ### 读取图片并进行QC
    output_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), os.path.splitext(os.path.basename(PATH))[0], 'cut')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### 读取图片
    WhiteImage = PATH
    image_path = WhiteImage
    cut_type = 2
    try:
        # img = cv.imread(WhiteImage, -1)
        # img = cv.imdecode(np.fromfile(WhiteImage, dtype=np.uint8), -1)
        img = tifi.imread(WhiteImage)
    except:
        try:
            img = tifi.imread(WhiteImage)
        except:
            img = cv.imdecode(np.fromfile(WhiteImage, dtype=np.uint8), -1)

    if cut_type == 2:
        if len(img.shape) == 3:
            print(img.shape)
            print('RGB2GRAY')
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            print(img.shape)
            # img = np.uint16(cv.normalize(img, None, np.min(img) * 255, np.max(img) * 255, cv.NORM_MINMAX))
            img = np.uint16(img) * 255
            print(img.shape)
            # magnification *= 1.3
        if img.dtype == 'uint16' and np.max(img) <= 255:
            img = img * 255

    magnification = si.ImageInfo.ScanObjective
    resize_shape, FOV_size, ori_size, mean_int = cut_image(img, magnification, os.path.splitext(os.path.basename(PATH))[0], output_dir, cut_type, si)

    row, col = get_row_col(output_dir)
    return output_dir, ori_size, [row, col], mean_int


def qc_net(PATH, si):
    small_path = PATH
    if si.ImageInfo.ScanRows is None or si.ImageInfo.ScanCols is None:
        rc_shape = get_row_col(small_path)
        si.ImageInfo.ScanRows = rc_shape[0] + 1
        si.ImageInfo.ScanCols = rc_shape[1] + 1
    # elif os.path.basename(PATH) == si.ImageInfo.SlideInfo or PATH.endswith('.imgs'):
    elif os.path.isdir(PATH):
        small_path = PATH
        if si.ImageInfo.ScanRows is None or si.ImageInfo.ScanCols is None:
            rc_shape = get_row_col(small_path)
            si.ImageInfo.ScanRows = rc_shape[0] + 1
            si.ImageInfo.ScanCols = rc_shape[1] + 1
        large_shape = [si.StitchInfo.ScopeStitch.GlobalWidth, si.StitchInfo.ScopeStitch.GlobalHeight]
    "初始化Info"
    info = FOVInfo(si, small_path)

    """图像清晰度算法"""
    iqa_start = time.time()
    # iqa = ImageQualityAssessment(si, info)
    # IQA_score, df_save_path, plot_save_path = iqa.predict()
    # si.UploadInfo.FileList.extend([df_save_path, plot_save_path])
    # # 转换为对应的分数 0-0.2为60分，0.2-0.8为80分, 0.8 - 1.0对应80-100分
    # if IQA_score <= 0.2:
    #     QCBlurScore = round(IQA_score * 60 / 0.2)
    # elif IQA_score <= 0.8:
    #     QCBlurScore = 60 + round(20 / 0.6 * (IQA_score - 0.2))
    # else:
    #     QCBlurScore = 80 + round(20 / 0.2 * (IQA_score - 0.8))
    QCBlurScore = -1

    si.QCInfo.QCBlurScore = QCBlurScore
    iqa_end = time.time()
    si.QCInfo.TimeCost.Iqa = int(iqa_end - iqa_start)

    """track点检测算法"""
    qn = fq.QualityControl(info, si)
    cal_ang, cal_scale, score, tc_str, cross_points, template_points = qn.qc()

    """更新结果到json"""
    si.QCInfo.Yolo = len(qn.track_cross_all)
    si.QCInfo.FovCount = info.fov_count
    end = time.time()
    si.QCInfo.TimeCost.Total = int(end - iqa_start)
    if PATH.endswith('.tif') or PATH.endswith('.png') or PATH.endswith('.czi'):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), os.path.splitext(os.path.basename(PATH))[0])
        rmtree(output_dir)

    return large_shape, cal_ang, cal_scale, score, tc_str, QCBlurScore, cross_points, template_points


