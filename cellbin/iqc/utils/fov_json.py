from utils import common
from constant import VERSION


# import common
class QcInfo(object):
    def __init__(self):
        pass

    def __getattr__(self, item):
        return None


class ChipInfo(QcInfo):
    def __init__(self):
        self.ChipID = 'DP84'
        self.Pitch = 715
        self.FOVTrackTemplate = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
                                 [112, 144, 208, 224, 224, 208, 144, 112, 160]]


class TimeCost(QcInfo):
    def __init__(self):
        self.Iqa = 0
        self.CrossPoint = 0
        self.LineDetect = 0
        self.TraditionalLineDetect = 0
        self.TemplateEval = 0


class QCInfo(QcInfo):
    def __init__(self):
        self.QCResultFlag = 1  # 0: failed, 1: success.
        self.QCScore = 0  # 90 is 90%.
        self.TrackFile = None
        self.QCVersion = VERSION
        self.TimeCost = TimeCost()


class RegisterInfo(QcInfo):
    def __init__(self):
        self.Scale = 1.0
        self.Rotation = 10000.0
        self.Barycenter = [1000, 1000]
        self.TrackInfo = 'track.xlsx'
        self.Offset = [0, 0]
        self.ChipRect = [100, 100, 1000, 1000]
        self.RegisterScore = 1.0


class BGIStitch(QcInfo):
    def __init__(self):
        self.PreStitchColPair = None  # [[3, 2], [2, 2]]
        self.PreStitchRowPair = None  # [[2, 3], [2, 2]]
        self.RawOffset = None  # 'raw_offset.xlsx'
        self.ValueOffset = None  # 'value_offset.xlsx'
        self.GlobalHeight = None  # 20000
        self.GlobalWidth = None  # 30000
        self.GlobalLoc = None  # bgi_location.xlsx


class ScopeStitch(QcInfo):
    def __init__(self):
        self.GlobalHeight = None  # 20000
        self.GlobalWidth = None  # 30000
        self.GlobalLoc = None  # scope_location.xlsx
        self.ScopeOffset = None  # scope_offset.xlsx


class StitchEval(QcInfo):
    def __init__(self):
        self.Flag = True  # True or False
        self.Rotation = None  # -0.1637017817341799
        self.Scale = None  # 0.7619047619047619
        self.TemplateSrc = None  # [3, 5]
        self.CredibleOffset = None  # credible_offset.xlsx
        self.GlobalHeight = None  # 20000
        self.GlobalWidth = None  # 30000
        self.GlobalLoc = None  # eval_location.txt
        self.GlobalTemplate = None  # global_template.txt
        self.GlobalDeviation = None   # 拼接误差矩阵
        self.MaxDeviation = None      # 最大拼接误差


class StitchInfo(QcInfo):
    def __init__(self):
        self.ScopeStitch = ScopeStitch()
        self.BGIStitch = BGIStitch()
        self.StitchEval = StitchEval()


class CellRecogInfo(QcInfo):
    def __init__(self):
        self.QCScore = 100
        self.GPU = -1
        self.Watershed = False


class ImageInfo(QcInfo):
    def __init__(self):
        self.Illuminance = None
        self.Gain = None
        self.TempPath = ''
        self.ImagePath = ''
        self.Manufacturer = 'motic'
        self.DeviceSN = None
        self.ControllerVer = 'Vers:LS21.00.038'
        self.CameraID = '\\\\?\\usb#vid_232f-pid_0105#6-36813388-0-2#REV1'
        self.CameraName = 'Moticam Pro 500i'
        self.ComputerName = 'DESKTOP-P3B5R9R'
        self.AppBuildDate = 'May 13 2021'
        self.AppName = 'PA53Scanner'
        self.AppFileVer = '1.0.0.3b'
        self.GUID = None
        self.ScanTime = None
        self.ScanObjective = None
        self.Scale = None
        self.BitDepth = 24
        self.Overlap = 0.1
        self.ScanRows = None
        self.ScanCols = None
        self.FOVHeight = None
        self.FOVWidth = None
        self.ExposureTime = 104.528587
        self.RGBScale = [1.0, 1.123147, 1.944359]
        self.Brightness = 50
        self.ColorEnhancement = 0
        self.Contrast = 0
        self.Gamma = 1.000000
        self.GammaShift = 0
        self.Sharpness = 0
        self.DistortionCorrection = False
        self.BackgroundBalance = False
        self.SlideInfo = ''
        self.Experimenter = ''
        self.RemarkInfo = ''
        self.stitched_image = False


class CellbinStitch(object):
    def __init__(self):
        self.ScopeStitch = ScopeStitch()
        self.BGIStitch = BGIStitch()
        self.StitchEval = StitchEval()


class CellbinQCInfo(object):
    def __init__(self, json_info):
        self.RemarkInfo = json_info.ImageInfo.RemarkInfo
        self.StainType = 'ssDNA'
        self.TrackDistanceTemplate = json_info.ChipInfo.FOVTrackTemplate
        self.QCPassFlag = json_info.QCInfo.QCResultFlag # 0: failed, 1: success.
        self.TrackLineScore = json_info.QCInfo.QCScore
        self.QCScore = 0  # 90 is 90%.
        self.CrossPoints = dict()
        self.GoodFOVCount = json_info.QCInfo.Yolo
        self.TotalFOVcount = json_info.QCInfo.FovCount
        self.ImageQCVersion = json_info.QCInfo.QCVersion
        self.Experimenter = json_info.ImageInfo.Experimenter


class CellbinImageInfo(object):
    def __init__(self, json_info):
        self.DistortionCorrection = json_info.ImageInfo.DistortionCorrection
        self.illuminance = json_info.ImageInfo.Illuminance
        self.Gain = json_info.ImageInfo.Gain
        self.Sharpness = json_info.ImageInfo.Sharpness
        self.Gamma = json_info.ImageInfo.Gamma
        self.GammaShift = json_info.ImageInfo.GammaShift
        self.RGBScale = json_info.ImageInfo.RGBScale
        self.WhiteBalance = json_info.ImageInfo.WhiteBalance
        self.BackgroundBalance = json_info.ImageInfo.BackgroundBalance

        self.Manufacturer = json_info.ImageInfo.Manufacturer
        self.DeviceSN = json_info.ImageInfo.DeviceSN
        self.ControllerVer = 'Vers:LS21.00.038'
        self.CameraID = '\\\\?\\usb#vid_232f-pid_0105#6-36813388-0-2#REV1'
        self.CameraName = 'Moticam Pro 500i'
        self.ComputerName = 'DESKTOP-P3B5R9R'
        self.AppBuildDate = 'May 13 2021'
        self.AppName = 'PA53Scanner'
        self.AppFileVer = '1.0.0.3b'
        self.GUID = None
        self.ScanTime = None
        self.ScanObjective = None
        self.Scale = None
        self.BitDepth = 24
        self.Overlap = 0.1
        self.ScanRows = None
        self.ScanCols = None
        self.FOVHeight = None
        self.FOVWidth = None
        self.ExposureTime = 104.528587
        self.RGBScale = [1.0, 1.123147, 1.944359]
        self.Brightness = 50
        self.ColorEnhancement = 0
        self.Contrast = 0

        self.Sharpness = 0

        self.BackgroundBalance = False
        self.SlideInfo = ''
        self.Experimenter = ''
        self.RemarkInfo = ''
        self.stitched_image = False


class Operation(QcInfo):
    def __init__(self):
        self.QCEval = True
        self.Stitching = True
        self.Register = True
        self.Segment = True
        self.Analysis = True


class UploadInfo(QcInfo):
    def __init__(self):
        self.MD5 = ''
        self.UploadCost = ''
        self.UploadTime = ''
        self.FileList = list()
        self.UploadStatus = 0


class AnalysisInfo(QcInfo):
    def __init__(self):
        self.SaveImgs = True
        self.SaveHtml = True


class ScopeInfo(object):
    def __init__(self):
        self.QCInfo = QCInfo()
        self.Operation = Operation()
        self.ImageInfo = ImageInfo()
        self.StitchInfo = StitchInfo()
        self.RegisterInfo = RegisterInfo()
        self.ChipInfo = ChipInfo()
        self.CellRecogInfo = CellRecogInfo()
        self.UploadInfo = UploadInfo()
        self.AnalysisInfo = AnalysisInfo()

    def serialize(self, file_path):
        dct = self.__dict__.copy()
        for attr in dct:
            if not hasattr(self, attr):
                delattr(self, attr)
        common.json_serialize(self, file_path)

    def deserialize(self, file_path):
        obj = common.json_deserialize(file_path)
        dct = self.__dict__.copy()
        for attr in dct:
            if hasattr(obj, attr):
                setattr(self, attr, getattr(obj, attr))
            # else:
            #     delattr(self, attr)


def main():
    # si = ScopeInfo()
    # si.serialize('E:\\Desktop\\scope_info.json')

    sj = ScopeInfo()
    sj.deserialize('/media/Data2/Image/stereomics/mouse_heart_eu/DP8400016861TR_C3/bgi_SS84.json')
    print(sj.ImageInfo.FOVHeight)
    print(sj.ImageInfo.ImagePath)
    print(sj.ImageInfo.TempPath)


if __name__ == '__main__':
    main()
