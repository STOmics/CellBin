#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import abspath, join, dirname
import glob
import datetime
import shutil
from re import search
from constant import DEBUG_MODE
import glog
import numpy as np
import QCs
from settings import Template
from utils.fov_json import ScopeInfo
from utils import common
from image_reader.read import motic_reader, big_image_reader
from cellbin_codec import ImageProcessRecord, set_cross_points

#########################################################
#########################################################
# Quality Control for Microscope Images
#########################################################
#########################################################

# Version and Date
PROG_VERSION = '0.2.0'
PROG_DATE = '2021-08-05'

# Usage
usage = '''

     Version %s  by Huang Zirui, Chen Bichao  %s

     Usage: %s -i <image_folder> [...]

''' % (PROG_VERSION, PROG_DATE, os.path.basename(sys.argv[0]))


class QualityControl(object):

    def __init__(self, input_path, report_str='',
                 output_path='', chip_name='',
                 exp_name='', extra_info='',
                 progress_type=1, upload_only=False, si=None, language='chn', cut_czi=False):

        self.input_path = input_path
        self.report_str = report_str
        self.chip_name = chip_name
        self.exp_name = exp_name
        self.extra_info = extra_info
        self.progress_type = progress_type
        self.upload_only = upload_only
        self.language = language
        self.cut_czi = cut_czi

        if len(output_path) == 0:
            self.output_path = os.path.join(dirname(abspath(get_path())), 'QCImgUpload')
        else:
            self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.output_file = ''

        if si is None:
            self.si = ScopeInfo()
            self.si_flag = False
        else:
            self.si = si
            self.si_flag = True
            motic_flag = 0
            for file in self.si.UploadInfo.FileList:
                if '1.ini' in file:
                    motic_flag = 1
                if motic_flag == 1:
                    break
            if motic_flag == 1:
                if not os.path.exists(os.path.join(self.input_path, '1.mdsx')):
                    fo = open(os.path.join(self.input_path, '1.mdsx'), 'w')
                    fo.close()
                if not os.path.exists(os.path.join(self.input_path, '3.jpg')):
                    fo = open(os.path.join(self.input_path, '3.jpg'), 'w')
                    fo.close()

    def analysis_net(self, score, image_shape, iqa):
        pass_thresh = 60

        if score >= pass_thresh:
            self.si.QCInfo.QCResultFlag = 1

        else:
            self.si.QCInfo.QCResultFlag = 0

        self.si.QCInfo.QCScore = score

    def run(self):

        if self.report_str != '':
            self.report_str += '\n'
        self.report_str += self.input_path + '\n'

        if os.path.isdir(self.input_path):
            self.ext = ""
        else:
            _, self.ext = os.path.splitext(self.input_path)
        if self.ext in ['.tif', '.png']:
            self.si.ImageInfo.ImagePath = self.input_path
            try:
                curr_dir = dirname(abspath(__file__))
                cfg_path = join(curr_dir, "", 'extraInfo.json')
                cfg_info = common.json_deserialize(cfg_path)
                glog.info(f"using extra info from {cfg_path}")
                # cfg_info = common.json_deserialize(os.path.join(abspath(get_path()), 'extraInfo.json'))
                self.read_config(cfg_info)
            except Exception as e:
                print(e)
                if self.language.lower() == 'chn':
                    self.report_str += '请正确填写额外信息。\n'
                elif self.language.lower() == 'chn':
                    self.report_str += 'Please input extra info corretly.\n'
                return -1
            try:
                br = big_image_reader.BigImageReader(
                    file_path=self.input_path,
                    save_path=self.output_path,
                    manufacturer=self.si.ImageInfo.Manufacturer,
                    chip_name=self.chip_name,
                    overlap=self.si.ImageInfo.Overlap,
                    fov_height=self.si.ImageInfo.FOVHeight,
                    fov_width=self.si.ImageInfo.FOVWidth,
                    scope_info=self.si
                )
                self.si = br.scope_info
                print("asd")
            except Exception as e:
                glog.info(Exception)

        elif self.ext == '':  # if input path is a folder
            # check if it is motic folder
            # if len(glob.glob(os.path.join(self.input_path, '*.mdsx'))) > 0:
            self.ext = '.mdsx'
            files = glob.glob(os.path.join(self.input_path, '*.*'))
            for i in range(len(files)):
                files[i] = files[i].replace("\\", "/")

            MR = motic_reader.MoticReader(self.input_path)
            if not self.si_flag:
                self.si = MR.scope_info
            else:
                MR.scope_info = self.si

            self.si.ImageInfo.ImagePath = os.path.join(self.input_path, '1.mdsx')
            if not os.path.exists(self.si.ImageInfo.TempPath.replace('.ini', '.imgs')):
                # name = os.path.splitext(os.path.basename(self.si.ImageInfo.TempPath.replace('\\', '/')))[0]
                name = os.path.splitext(os.path.basename(self.si.ImageInfo.TempPath.replace('\\', '/')))[0]
                # 默认为从info信息中获取的temppath
                im_d = os.path.join(self.input_path, name + '.imgs')
                if not os.path.exists(im_d):
                    im_d = os.path.join(self.input_path,
                                        os.path.basename(self.si.ImageInfo.TempPath.replace('\\', '/')))
                if not os.path.exists(im_d):
                    if len(self.si.ImageInfo.SlideInfo) != 0:
                        im_d = os.path.join(os.path.dirname(self.si.ImageInfo.ImagePath),
                                            self.si.ImageInfo.SlideInfo)
                if not os.path.exists(im_d):
                    im_d = os.path.join(self.input_path, self.chip_name)

                if not os.path.exists(im_d):
                    im_d = os.path.join(self.input_path, self.si.ImageInfo.SlideInfo)

                if not os.path.exists(im_d):
                    im_d = os.path.join(os.path.dirname(self.si.ImageInfo.ImagePath),
                                        os.path.basename(os.path.dirname(self.si.ImageInfo.ImagePath)))
                if not os.path.exists(im_d):
                    im_d = os.path.join(self.input_path, os.path.basename(self.input_path))
                if os.path.exists(im_d):
                    self.si.ImageInfo.TempPath = im_d
                else:
                    if self.language.lower() == 'chn':
                        self.report_str += '小图文件夹不存在，请检查对应路径。\n'
                    elif self.language.lower() == 'eng':
                        self.report_str += 'Fov images folder does not exist, please check corresponding path.\n'
                    return -1

        ret = self.check_info()
        if ret == -1:
            return -1

        times = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_name = "_".join([self.chip_name, times, self.si.QCInfo.QCVersion])
        self.si.ImageInfo.RemarkInfo = self.extra_info
        self.si.QCInfo.OutputName = output_name

        self.output_file = os.path.join(self.output_path, output_name + '.json')
        if os.path.exists(self.output_file):
            if os.path.exists(self.output_file.replace('.json', '.tar.gz')):
                exist_si = common.json_deserialize(self.output_file)
                if exist_si.QCInfo.QCResultFlag == 1:
                    exist_si.UploadInfo.UploadStatus = 1
                    common.json_serialize(exist_si, self.output_file)
                    return 0

        if self.ext == '.czi':
            self.si.ImageInfo.TempPath = os.path.join(self.output_path, self.chip_name)

        if self.upload_only:
            self.si.QCInfo.QCResultFlag = -1
            self.si.Operation.QCEval = True
        else:
            tc_str, error_flag, self.trackline_score, cross_points, template_points = self.get_result()
            self.si.Operation.QCEval = False

        if not self.upload_only:
            with open(join(self.output_path, output_name + '.txt'), 'w') as fw:
                fw.writelines(tc_str)
            self.si.QCInfo.TrackFile = output_name + '.txt'
        stitch_loc = None
        if self.ext == '.mdsx':
            stitch_loc = MR.stitch_info.loc
            MR.generate_xlsx(self.output_path, output_name)
        elif self.ext in ['.tif', '.png']:
            stitch_loc = br.stitch_info.loc
            br.generate_xlsx(self.output_path, output_name)

        if not self.upload_only:
            pass
        self.si.ImageInfo.ImagePath = os.path.dirname(self.si.ImageInfo.ImagePath)

        self.si.serialize(self.output_file)

        image_process_record = ImageProcessRecord(self.si)
        if stitch_loc is not None:
            image_process_record.Stitch.ScopeStitch.GlobalLoc = stitch_loc
        image_process_record.to_file(join(self.output_path, output_name + '.ipr'))
        tranformed_cross_points = {}
        for row_column, points in cross_points.items():
            # print(points)
            if points:
                points_array = [[point[0][0], point[0][1], 0, 0] for point in points]
                tranformed_cross_points[row_column] = points_array
        set_cross_points(join(self.output_path, output_name + '.ipr'), tranformed_cross_points, template_points)

        if error_flag is True:
            if self.language.lower() == 'chn':
                self.report_str += 'QC过程存在异常，请联系FAS。\n'
            elif self.language.lower() == 'eng':
                self.report_str += 'Thers is something error in qc pipeline，Please contact FAS.\n'
        else:
            if self.language.lower() == 'chn':
                self.report_str += 'QC完成。\n'
            elif self.language.lower() == 'eng':
                self.report_str += 'QC finished.\n'

        return 0

    def get_result(self):
        if self.ext in ['.tif', '.png']:
            img_dir = self.si.ImageInfo.TempPath
        elif self.ext == '.mdsx':
            img_dir = self.si.ImageInfo.TempPath.replace('.ini', '.imgs')
        elif self.ext == '.czi':
            if self.cut_czi:
                img_dir = self.si.ImageInfo.ImagePath
            else:
                img_dir = self.si.ImageInfo.TempPath
        elif self.ext == "xml":
            img_dir = self.si.ImageInfo.ImagePath

        try:
            self.si.QCInfo.OutputPath = self.output_path
            self.si.QCInfo.ChipName = self.chip_name
            self.si.QCInfo.Debug = DEBUG_MODE
            glog.info(f"Debug -> {self.si.QCInfo.Debug}")
            glog.info(f"FOV path -> {img_dir}")
            image_shape, rot_ang, scale, score, tc_str, IQA_score, cross_points, template_points = QCs.qc_net(img_dir,
                                                                                                              self.si)
            error_flag = False
        except Exception as e:
            print('=== STEP ERROR INFO START')
            import traceback
            traceback.print_exc()
            print('=== STEP ERROR INFO END')
            error_flag = True
            image_shape = np.array([0, 0])
            QCresult_bad = np.array([[81], [81]])
            QCresult_tot = np.array([[81], [81]])
            rot_ang = 0
            scale = 1
            tc_str = ''
            if self.si_flag:
                scan_path = os.path.join(self.output_path, 'qc_temp')
            else:
                scan_path = os.path.join(os.path.abspath(get_path()), os.path.splitext(os.path.basename(img_dir))[0])
            if os.path.exists(scan_path):
                pass
            raise e
        self.analysis_net(score, image_shape, IQA_score)
        self.si.RegisterInfo.Rotation = rot_ang
        self.si.RegisterInfo.Scale = scale
        self.si.ImageInfo.TempPath = self.si.ImageInfo.TempPath.replace('.ini', '.imgs')
        return tc_str, error_flag, score, cross_points, template_points

    def check_info(self):
        chipCheck = 0
        chipNo = ''
        if self.chip_name == '':
            for chip in Template.keys():
                if self.si.ImageInfo.SlideInfo is None:
                    chipCheck = 0
                # elif not self.si.ImageInfo.SlideInfo.startswith(chip):
                #     chipCheck = 0
                elif not check_chip_no(self.si.ImageInfo.SlideInfo):
                    chipCheck = 0
                else:
                    chipCheck = 1
                    chipNo = chip
                    break
            if chipCheck == 0:
                dir = os.path.basename(self.si.ImageInfo.ImagePath)
                for chip in Template.keys():
                    if not dir.startswith(chip):
                        chipCheck = 0
                    else:
                        chipCheck = 1
                        chipNo = chip
                        break
                if chipCheck == 0:
                    if self.progress_type == 1 or self.progress_type == 3:
                        if self.language.lower() == 'chn':
                            self.report_str += '未找到符合标准的芯片号，请输入芯片号。\n'
                        elif self.language.lower() == 'eng':
                            self.report_str += 'Cannot find chip number with standard format，Please input chip number.\n'
                        return -1
                    else:
                        if self.language.lower() == 'chn':
                            self.report_str += '未找到芯片号，跳过该文件夹。\n'
                        elif self.language.lower() == 'eng':
                            self.report_str += 'Cannot find chip number，Skip current folder.\n'
                        return -1
                else:
                    self.chip_name = dir
                    self.si.ImageInfo.SlideInfo = self.chip_name
            else:
                self.chip_name = self.si.ImageInfo.SlideInfo
        else:
            if not check_chip_no(self.chip_name):
                if self.language.lower() == 'chn':
                    self.report_str += '不支持当前输入的芯片号，请重新输入。\n'
                    self.report_str += '支持的芯片号：' + ', '.join(Template.keys())
                elif self.language.lower() == 'eng':
                    self.report_str += 'Chip number input is not supported，Please retry.\n'
                    self.report_str += 'Supported chip numbers: ' + ', '.join(Template.keys())
                return -1
            else:
                self.si.ImageInfo.SlideInfo = self.chip_name
                for chip in Template.keys():
                    if chip in self.chip_name:
                        chipNo = chip
                    else:
                        chipNo = "SS2"
                        break

        self.si.ChipInfo.ChipID = chipNo
        self.si.ChipInfo.Pitch = Template[chipNo]["pitch"]
        self.si.ChipInfo.FOVTrackTemplate = [Template[chipNo]["grids"], Template[chipNo]["grids"]]

        if self.exp_name == '':
            if not self.si.ImageInfo.Experimenter:
                if self.language.lower() == 'chn':
                    self.report_str += '请输入实验员邮箱前缀。\n'
                elif self.language.lower() == 'eng':
                    self.report_str += 'Please input E-mail address of experimenter. \n'
                return -1
        else:
            if '@' in self.exp_name:
                self.si.ImageInfo.Experimenter = self.exp_name
            else:
                self.si.ImageInfo.Experimenter = self.exp_name + '@genomics.cn'

        return 0

    def read_config(self, cfg):
        self.si.ImageInfo.Manufacturer = cfg.Info.Manufacturer
        self.si.ImageInfo.ScanObjective = float(cfg.Info.ScanObjective)
        self.si.ImageInfo.Scale = float(cfg.Info.Scale)
        self.si.ImageInfo.Overlap = float(cfg.Info.Overlap)
        self.si.ImageInfo.FOVWidth = int(cfg.Info.FOVWidth)
        self.si.ImageInfo.FOVHeight = int(cfg.Info.FOVHeight)

        self.si.RegisterInfo.Rotation = float(cfg.Info.Angle)

        chipNo = cfg.Info.Chip
        # self.si.ImageInfo.SlideInfo = chipNo
        self.si.ChipInfo.ChipID = chipNo
        self.si.ChipInfo.Pitch = Template[chipNo]["pitch"]
        self.si.ChipInfo.FOVTrackTemplate = [Template[chipNo]["grids"], Template[chipNo]["grids"]]


def check_chn(str):
    chn = r'[\u4e00-\u9fa5]+'
    match = search(chn, str)
    if match:
        return True
    else:
        return False


def get_path():
    if hasattr(sys, 'frozen'):
        return dirname(sys.executable)
    else:
        return os.getcwd()


def qc_entry_app(*args):
    """ Entry for AppQC. """
    qual = QualityControl(*args)
    ret = qual.run()
    if ret == -1:
        return qual.report_str
    elif ret == 0:
        return qual.report_str, qual.output_path, qual.output_file, qual.trackline_score


def qc_entry(para):
    """ Entry for function calling. """
    # Initialize quality control class
    qual = QualityControl(**vars(para))
    # run quality control
    ret = qual.run()
    if ret == -1:
        print(qual.report_str)
        return -1
    else:
        report_str = qual.report_str, qual.output_path, qual.output_file
        from utils.common import json_serialize, json_deserialize
        if isinstance(report_str, tuple):
            output_path = report_str[1]
            output_file = report_str[2]
            report_str = report_str[0]
            json_info = json_deserialize(output_file)
    return 0, qual.output_file


def check_slide_no(slide_no):
    tails = ["TL", "TR", "BL", "BR"]
    if len(slide_no) == 13 and slide_no[:3] in ("SS2", "DP8", "FP2"):
        if slide_no[3:11].isdigit() and slide_no[11:13] in tails:
            return True
    elif len(slide_no) == 14 and slide_no[:4] == "DP84":
        if slide_no[4:12].isdigit() and slide_no[12:14] in tails:
            return True
    elif len(slide_no) == 11 and slide_no[:3] in ("FP2", "SS2") and slide_no[3:11].isdigit():
        return True
    return False


s13_shard_set = ("ABCDEFGHJKLMN", "123456789ABCD")
s6_shard_set = ("ABCDEF", "123456")


def check_shard_no(shard_no, slide_len):
    if slide_len == 11:
        shard_set = s13_shard_set
    else:
        shard_set = s6_shard_set
    if len(shard_no) == 2:
        return shard_no[0] in shard_set[0] and shard_no[1] in shard_set[1]
    elif len(shard_no) == 4:
        return shard_no[0] in shard_set[0] and shard_no[1] in shard_set[1] and (
                shard_no[2] in shard_set[0] and shard_no[3] in shard_set[1])
    return False


def check_short_no(chip_no):
    if chip_no[1:6].isdigit():
        if chip_no[0] in 'ABCDY' and chip_no[6:] == '00':
            return True
        elif chip_no[0] in 'ABCD':
            return chip_no[6] in s6_shard_set[0] and chip_no[7] in s6_shard_set[1]
        elif chip_no[0] == "Y":
            return chip_no[6] in s13_shard_set[0] and chip_no[7] in s13_shard_set[1]
        else:
            return False
    return False


def check_chip_no(chip_no):
    """
    长码规则：
    s6(6cm*6cm)DipT10-V8的命名规则  起始符（SS/FP) + 2(版本相关) + xxxxxxxx(8位流水号) +  尾号（TL/TR/BL/BR）+ 区域号（A1-F6, 可能是两位，也可能是四位）
    s6(6cm*6cm)DipT1的命名规则  起始符（DP8) + 4或空(版本相关) + xxxxxxxx(8位流水号) +  尾号（TL/TR/BL/BR）+ 区域号（A1-F6, 可能是两位，也可能是四位）
    S13(13cm*13cm) DipT10的命名规则  起始符（SS/FP) + 2(版本相关) + xxxxxxxx(8位流水号) + 区域号（A1-ND, 可能是两位，也可能是四位）
    注意：所有的字母都是大写

    短码规则：
    T10短码规则 芯片方向（ABCDY） + XXXXX（5位流水号） + 切割的小芯片（不切割为00）
    """

    split_ = chip_no.split('_')
    if len(split_) >= 2 and len(split_[0]) != 8:
        slide_no, shard_no = split_[:2]
        result = check_slide_no(slide_no) and check_shard_no(shard_no, len(slide_no))
    elif len(split_[0]) == 8:
        result = check_short_no(split_[0])
    else:
        return False
    return result


def main():
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage)
    ArgParser.add_argument("--version", action="version", version=PROG_VERSION)
    ArgParser.add_argument("-i", "--input", action="store", dest="input_path", type=str, required=True,
                           help="Image folder.")
    ArgParser.add_argument("-o", "--output", action="store", dest="output_path", type=str, default='',
                           help="QC results output folder.")
    ArgParser.add_argument("-c", "--chip", action="store", dest="chip_name", type=str, default='FP2', help="Chip name")
    ArgParser.add_argument("-n", "--name", action="store", dest="exp_name", type=str, default='',
                           help="Name of the experimenter.")
    ArgParser.add_argument("-e", "--info", action="store", dest="extra_info", type=str, default='',
                           help="Extra information.")

    (para, args) = ArgParser.parse_known_args()

    ret = qc_entry(para)
    if ret == -1:
        print("QC exit with error!")
        sys.exit(-1)


if __name__ == "__main__":
    main()
