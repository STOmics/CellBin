import glob
import os
from h5py import File
import numpy
from utils import common
from copy import deepcopy
# vipshome = os.path.dirname(os.path.abspath(__file__)) + r'\vips-dev-w64-all-8.12.2\vips-dev-8.12\bin'
# os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
# import pyvips


class Celbin(object):
    def __init__(self):
        pass

    def deserialize_to_h5(self, obj, group):
        dct = obj.__dict__.copy()
        for attr, v in dct.items():
            if isinstance(v, numpy.ndarray):
                group.create_dataset(attr, data=v)
            elif isinstance(v, list):
                group.create_dataset(attr, data=numpy.array(v))
            elif isinstance(v, Celbin):
                info_group = group.create_group(attr, track_order=True)
                self.deserialize_to_h5(v, info_group)
            else:
                if v is None:
                    v = '-'
                group.attrs[attr] = v


class ImageProcessRecord(Celbin):
    def __init__(self, json_file):
        if isinstance(json_file, str):
            json_info = common.json_deserialize(json_file)
        else:
            json_info = json_file
        self.QCInfo = CellbinQCInfo(json_info)
        self.ImageInfo = CellbinImageInfo(json_info)
        self.Stitch = CellbinStitch(json_info)
        self.Register = CellbinRegister(json_info)

    def to_file(self, file_path):
        with File(file_path, 'w') as f:
            self.deserialize_to_h5(self, f)


class CellbinRegister(Celbin):
    def __init__(self, json_info):
        self.ScaleX = json_info.RegisterInfo.Scale
        self.ScaleY = json_info.RegisterInfo.Scale
        self.Rotation = json_info.RegisterInfo.Rotation


class ScopeStitch(Celbin):
    def __init__(self, json_info):
        self.GlobalHeight = json_info.StitchInfo.ScopeStitch.GlobalHeight
        self.GlobalWidth = json_info.StitchInfo.ScopeStitch.GlobalWidth
        self.GlobalLoc = None  # scope_location.xlsx


class StitchEval(Celbin):
    def __init__(self, json_info):
        self.MaxDeviation = json_info.StitchInfo.StitchEval.MaxDeviation
        self.GlobalDeviation = json_info.StitchInfo.StitchEval.GlobalDeviation


class CellbinStitch(Celbin):
    def __init__(self, json_info):
        self.ScopeStitch = ScopeStitch(json_info)
        if hasattr(json_info.ImageInfo, 'stitched_image'):
            if json_info.ImageInfo.stitched_image:
                self.StitchEval = StitchEval(json_info)
            else:
                self.TemplateSource = json_info.StitchInfo.StitchEval.TemplateSrc
                if isinstance(self.TemplateSource, list):
                    self.TemplateSource = '_'.join(map(str, self.TemplateSource))


class CellbinQCInfo(Celbin):
    def __init__(self, json_info):
        self.ClarityScore = json_info.QCInfo.QCBlurScore
        self.RemarkInfo = json_info.ImageInfo.RemarkInfo
        self.StainType = 'ssDNA'
        self.TrackDistanceTemplate = json_info.ChipInfo.FOVTrackTemplate
        self.QCPassFlag = json_info.QCInfo.QCResultFlag  # 0: failed, 1: success.
        self.TrackLineScore = json_info.QCInfo.QCScore
        self.GoodFOVCount = json_info.QCInfo.Yolo
        self.TotalFOVcount = json_info.QCInfo.FovCount
        self.ImageQCVersion = json_info.QCInfo.QCVersion
        self.Experimenter = json_info.ImageInfo.Experimenter


class CellbinImageInfo(Celbin):
    def __init__(self, json_info):
        self.DistortionCorrection = json_info.ImageInfo.DistortionCorrection
        self.Illuminance = json_info.ImageInfo.Illuminance
        self.Gain = json_info.ImageInfo.Gain
        self.Sharpness = json_info.ImageInfo.Sharpness
        self.Gamma = json_info.ImageInfo.Gamma
        self.GammaShift = json_info.ImageInfo.GammaShift
        self.Contrast = json_info.ImageInfo.Contrast
        self.RGBScale = json_info.ImageInfo.RGBScale
        self.WhiteBalance = json_info.ImageInfo.WhiteBalance
        self.BackgroundBalance = json_info.ImageInfo.BackgroundBalance
        self.ColorEnhancement = json_info.ImageInfo.ColorEnhancement
        self.ChannelCount = json_info.ImageInfo.Channel
        self.STOmicsChipSN = json_info.ImageInfo.SlideInfo
        self.Manufacturer = json_info.ImageInfo.Manufacturer
        self.Model = json_info.ImageInfo.CameraName
        self.ScanRows = json_info.ImageInfo.ScanRows
        self.ScanCols = json_info.ImageInfo.ScanCols
        self.FOVHeight = json_info.ImageInfo.FOVHeight
        self.FOVWidth = json_info.ImageInfo.FOVWidth
        self.BitDepth = json_info.ImageInfo.BitDepth
        self.Overlap = json_info.ImageInfo.Overlap
        self.ScanObjective = json_info.ImageInfo.ScanObjective
        if hasattr(json_info.ImageInfo, 'PixelSizeX'):
            self.PixelSizeX = json_info.ImageInfo.PixelSizeX
            self.PixelSizeY = json_info.ImageInfo.PixelSizeY
        else:
            self.PixelSizeX = json_info.ImageInfo.Scale
            self.PixelSizeY = json_info.ImageInfo.Scale
        self.StitchedImage = json_info.ImageInfo.stitched_image
        self.QCResultFile = json_info.QCInfo.OutputName
        self.DeviceSN = json_info.ImageInfo.DeviceSN
        self.ScanChannel = json_info.ImageInfo.ScanChannel
        self.ScanTime = json_info.ImageInfo.ScanTime
        self.ExposureTime = json_info.ImageInfo.ExposureTime
        self.Brightness = json_info.ImageInfo.Brightness
        self.AppFileVer = json_info.ImageInfo.AppFileVer


def set_cross_points(h5file, cross_points, template_source):
    with File(h5file, 'r+') as f:
        Stitch = f.require_group('Stitch')
        QCInfo = f.require_group('QCInfo')
        CrossPoints = QCInfo.create_group('CrossPoints', track_order=True)
        for row_column, points in template_source.items():
            Stitch.attrs['TemplateSource'] = row_column
            CrossPoints.create_dataset(row_column, data=numpy.array(points[1]))

        for row_column, points in cross_points.items():
            if points:
                CrossPoints.create_dataset(row_column, data=numpy.array(points))


def read_points_to_array(cross_points_file):

    with open(cross_points_file, 'r') as f:
        cross_points = {}
        template_source = {}
        for line in f.readlines():
            items = line.strip('\n').split('\t')
            if '_' in line:
                index = items[0]
                if not template_source and len(cross_points.keys()) == 1:
                    template_source = deepcopy(cross_points)
                    cross_points = {}
                cross_points[index] = []
            elif not line.isspace():
                cross_points[index].append([float(i) for i in items])
    return cross_points, template_source


def trans_json_ipr(json_file, stitch_file, points_file):
    out_path = json_file.replace('.json', '.ipr')
    image_process_record = ImageProcessRecord(json_file)
    # 保存拼接坐标
    stitch_loc = common.location2npy(stitch_file)
    image_process_record.Stitch.ScopeStitch.GlobalLoc = stitch_loc
    image_process_record.to_file(out_path)
    # 保存track点
    cross_points, template_source = read_points_to_array(points_file)
    set_cross_points(out_path, cross_points, template_source)


def main(input_path):
    json_file = glob.glob(os.path.join(input_path, '*.json'))[0]
    stitch_file = glob.glob(os.path.join(input_path, '*.xlsx'))[0]
    points_file = glob.glob(os.path.join(input_path, '*.txt'))[0]
    trans_json_ipr(json_file, stitch_file, points_file)


if __name__ == "__main__":
    input_path = r"D:\work\ImageDataBase\cellbin_jx\SS200000150BR_D4_20220418_142731_1.0.7"
    main(input_path)

