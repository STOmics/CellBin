import os.path
import struct
import glog
import numpy as np
from os.path import join, exists
from .fov_reader import FOVReader
from configparser import ConfigParser
import glob


class UnitAssemblyInfo(object):
    def __init__(self):
        self.row: int = 0
        self.col: int = 0
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.focusfactor: float = 0.0
        self.leftDx: int = 0
        self.leftDy: int = 0
        self.topDx: int = 0
        self.topDy: int = 0
        self.rightDx: int = 0
        self.rightDy: int = 0
        self.bottomDx: int = 0
        self.bottomDy: int = 0
        self.linkLeft: bool = False
        self.linkRight: bool = False
        self.linkTop: bool = False
        self.linkBottom: bool = False

    def set_from_tuple(self, tpl):
        dct = self.__dict__.copy()
        ind = 0
        for attr in dct:
            setattr(self, attr, tpl[ind])
            ind += 1

    def print_info(self, header=True):
        dct = self.__dict__.copy()
        if header:
            print(dct.keys())
        else:
            print(dct.values())


class MoticReader(FOVReader):
    def __init__(self, file_path):
        super(MoticReader, self).__init__(file_path)
        self.file_path = file_path
        self.manufacturer = 'motic'
        ini_1 = join(file_path, '1.ini')
        ini_info = join(file_path, 'info.ini')
        if exists(ini_info):
            self.read_info_ini(ini_info)
            self.scope_info.ImageInfo.ImagePath = file_path
            self.scope_info.ImageInfo.Manufacturer = self.manufacturer
        else:
            glog.warn('File {} does not exist.'.format(ini_info))
            dirs = glob.glob('{}\\*{}'.format(file_path, '.imgs'))
            assert len(dirs) == 1
            imgs_path = join(file_path,  dirs[0])
            self.set_scope_info_from_imgs(imgs_path)

        if exists(ini_1):
            self.read_1ini(ini_1)
        else:
            glog.warn('File {} does not exist.'.format(ini_1))

    def read_mdsx(self, ):
        self.stitch_info = None
        glog.info('Does not support reading data from motic source files currently.')

    def read_1ini(self, ini_path):
        with open(ini_path, 'rb') as fd:
            # header, self.fov_width, self.fov_height, self.bg_color = struct.unpack('<4s3i', fd.read(16))
            header, w_, h_, self.bg_color = struct.unpack('<4s3i', fd.read(16))
            rect = struct.unpack('<4i', fd.read(16))
            self.stitch_info.global_height = rect[3]
            self.stitch_info.global_width = rect[2]
            glog.info('FOV shape: ({}, {}), mosaic shape: {}.'.format(self.fov_height,
                                                                      self.fov_width, (rect[3], rect[2])))
            # self.fov_rows, self.fov_cols = struct.unpack('<2i', fd.read(8))
            r_, c_ = struct.unpack('<2i', fd.read(8))
            glog.info('Block count as, {} X {}.'.format(self.fov_rows, self.fov_cols))
            self.stitch_info.loc = np.zeros((self.fov_rows, self.fov_cols, 2), dtype=int)

            for i in range(self.fov_rows):
                for j in range(self.fov_cols):
                    v = struct.unpack('<2i4d8i4?', fd.read(76))
                    uai = UnitAssemblyInfo()
                    uai.set_from_tuple(v)
                    # uai.print_info(header=False)
                    fd.read(4)
                    self.stitch_info.loc[i, j] = [int(uai.x), int(uai.y)]
            # self.stitch_info.loc = self.stitch_info.loc.tolist()
            self.stitch_info.overlap = int.from_bytes(fd.read(4), 'little')
            glog.info('Overlap is {}%%.'.format(self.stitch_info.overlap))

        self.scope_info.StitchInfo.ScopeStitch.GlobalWidth = self.stitch_info.global_width
        self.scope_info.StitchInfo.ScopeStitch.GlobalHeight = self.stitch_info.global_height

    @staticmethod
    def get_value(cp, sec, op):
        if cp.has_option(sec, op):
            return cp.get(sec, op)
        else:
            glog.warn('Section {} missing option {}.'.format(sec, op))
            return None

    def init_by_version_1d0d0d3b(self, cp):
        self.scope_info.ImageInfo.TempPath = self.get_value(cp, 'AssemblyInfo', 'TempPath').replace('.ini', '.imgs')
        """ Assigned:
                ImagePath, Manufacturer
        """
        self.scope_info.ImageInfo.DeviceSN = self.get_value(cp, 'info', 'DeviceSN')
        """ Miss:
                ControllerVer, CameraID, CameraName, ComputerName, AppBuildDate, AppName
        """
        # Assigned: AppFileVer
        self.scope_info.ImageInfo.GUID = self.get_value(cp, 'info', 'GUID')
        self.scope_info.ImageInfo.ScanTime = self.get_value(cp, 'info', 'createTimeText')  # createTimeText
        self.scope_info.ImageInfo.ScanObjective = float(self.get_value(cp, 'AssemblyInfo', 'lens'))
        self.scope_info.ImageInfo.Scale = float(self.get_value(cp, 'info', 'scale'))
        # Miss: BitDepth
        self.scope_info.ImageInfo.Overlap = float(self.get_value(cp, 'AssemblyInfo', 'Overlap'))
        self.scope_info.ImageInfo.ScanRows = int(self.get_value(cp, 'AssemblyInfo', 'Row'))
        self.scope_info.ImageInfo.ScanCols = int(self.get_value(cp, 'AssemblyInfo', 'Col'))
        self.scope_info.ImageInfo.FOVHeight = int(self.get_value(cp, 'info', 'DevVideoHeight'))
        self.scope_info.ImageInfo.FOVWidth = int(self.get_value(cp, 'info', 'DevVideoWidth'))
        """ Miss:
                ExposureTime, RGBScale, Brightness, ColorEnhancement, Contrast, Gamma, GammaShift, 
                Sharpness, DistortionCorrection, BackgroundBalance, SlideInfo, Experimenter
        """

    def init_by_version_1d0d0d7b(self, cp):
        self.scope_info.ImageInfo.TempPath = self.get_value(cp, 'AssemblyInfo', 'TempPath').replace('.ini', '.imgs')
        """ Assigned:
                ImagePath, Manufacturer
        """
        self.scope_info.ImageInfo.DeviceSN = self.get_value(cp, 'info', 'DeviceSN')
        self.scope_info.ImageInfo.ControllerVer = self.get_value(cp, 'Property', 'sys.ControllerVer')
        self.scope_info.ImageInfo.CameraID = self.get_value(cp, 'Property', 'sys.CameraID')
        self.scope_info.ImageInfo.CameraName = self.get_value(cp, 'Property', 'sys.CameraName')
        self.scope_info.ImageInfo.ComputerName = self.get_value(cp, 'Property', 'sys.ComputerName')
        self.scope_info.ImageInfo.AppBuildDate = self.get_value(cp, 'Property', 'sys.AppBuildDate')
        self.scope_info.ImageInfo.AppName = self.get_value(cp, 'Property', 'sys.AppName')
        # Assigned: AppFileVer
        self.scope_info.ImageInfo.GUID = self.get_value(cp, 'info', 'GUID')
        self.scope_info.ImageInfo.ScanTime = self.get_value(cp, 'info', 'createTimeText')  # createTimeText
        self.scope_info.ImageInfo.ScanObjective = float(self.get_value(cp, 'AssemblyInfo', 'lens'))
        self.scope_info.ImageInfo.Scale = float(self.get_value(cp, 'info', 'scale'))
        # Need modify: BitDepth
        self.scope_info.ImageInfo.BitDepth = \
            int(len(bin(int(self.get_value(cp, 'Property', 'video.MaxGreyLevel')))) - 2)
        self.scope_info.ImageInfo.Overlap = float(self.get_value(cp, 'AssemblyInfo', 'Overlap'))
        self.scope_info.ImageInfo.ScanRows = int(self.get_value(cp, 'AssemblyInfo', 'Row'))
        self.scope_info.ImageInfo.ScanCols = int(self.get_value(cp, 'AssemblyInfo', 'Col'))
        self.scope_info.ImageInfo.FOVHeight = int(self.get_value(cp, 'info', 'DevVideoHeight'))
        self.scope_info.ImageInfo.FOVWidth = int(self.get_value(cp, 'info', 'DevVideoWidth'))
        self.scope_info.ImageInfo.ExposureTime = float(self.get_value(cp, 'Property', 'video.Exposure'))
        red_scale = float(self.get_value(cp, 'Property', 'video.RedScale'))
        green_scale = float(self.get_value(cp, 'Property', 'video.GreenScale'))
        blue_scale = float(self.get_value(cp, 'Property', 'video.BlueScale'))
        self.scope_info.ImageInfo.RGBScale = [red_scale, green_scale, blue_scale]
        self.scope_info.ImageInfo.Brightness = int(self.get_value(cp, 'Property', 'video.Brightness'))
        self.scope_info.ImageInfo.ColorEnhancement = bool(int(self.get_value(cp, 'Property', 'video.ColorEnhancement')))
        self.scope_info.ImageInfo.Contrast = bool(int(self.get_value(cp, 'Property', 'video.Contrast')))
        self.scope_info.ImageInfo.Gamma = float(self.get_value(cp, 'Property', 'video.Gamma'))
        self.scope_info.ImageInfo.GammaShift = bool(int(self.get_value(cp, 'Property', 'video.GammaShift')))
        self.scope_info.ImageInfo.Sharpness = bool(int(self.get_value(cp, 'Property', 'video.Sharpness')))
        self.scope_info.ImageInfo.DistortionCorrection = bool(int(self.get_value(cp, 'Property', 'video.Distort')))
        self.scope_info.ImageInfo.BackgroundBalance = bool(int(self.get_value(cp, 'Property', 'video.Background')))
        self.scope_info.ImageInfo.SlideInfo = self.get_value(cp, 'Property', 'bgi.ChipNo')
        self.scope_info.ImageInfo.Experimenter = self.get_value(cp, 'Property', 'bgi.Experimenter')
        self.scope_info.ImageInfo.Gain = self.get_value(cp, 'Property', 'Gain')
        self.scope_info.ImageInfo.ScanChannel = self.get_value(cp, 'Property', 'microscope.LightPath')

    def read_info_ini(self, ini_path):
        cp = ConfigParser()
        cp.read(ini_path, encoding='utf-8')

        app_file_ver = self.get_value(cp, 'info', 'ScanMachineInfo').split(' ')[1]
        # assert app_file_ver in ['1.0.0.3b', '1.0.0.7b', '1.0.5.132b', '1.0.0.8b']
        glog.info('Version of info file is {}.'.format(app_file_ver))
        if app_file_ver == '1.0.0.3b': self.init_by_version_1d0d0d3b(cp)
        elif app_file_ver == '1.0.0.7b': self.init_by_version_1d0d0d7b(cp)
        elif app_file_ver == '1.0.5.132b': self.init_by_version_1d0d0d3b(cp)
        elif app_file_ver == '1.0.0.8b': self.init_by_version_1d0d0d7b(cp)
        else:
            glog.info('Unsupported version type.')
            self.init_by_version_1d0d0d7b(cp)
        self.scope_info.ImageInfo.AppFileVer = app_file_ver

        self.fov_cols = int(self.get_value(cp, 'AssemblyInfo', 'Col'))
        self.fov_rows = int(self.get_value(cp, 'AssemblyInfo', 'Row'))
        self.fov_width = int(self.get_value(cp, 'info', 'DevVideoWidth'))
        self.fov_height = int(self.get_value(cp, 'info', 'DevVideoHeight'))
        self.fov_count = self.fov_cols * self.fov_rows
