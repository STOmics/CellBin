import common


class QcInfo(object):
    def __init__(self):
        pass

    def __getattr__(self, item):
        return None


class ChipInfo(QcInfo):
    def __init__(self):
        self.ChipID = 'FP2'
        self.Pitch = 715
        self.FOVTrackTemplate = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
                                 [112, 144, 208, 224, 224, 208, 144, 112, 160]]


class QCInfo(QcInfo):
    def __init__(self):
        self.QCResultFlag = 1  # 0: failed, 1: success.
        self.GoodFOVCount = 0
        self.QCScore = 0       # 90 is 90%.
        self.HeatMap = '.csv'
        self.TrackFile = None


class RegisterInfo(QcInfo):
    def __init__(self):
        self.Scale = 1.0
        self.Rotation = 0.0
        self.Barycenter = [1000, 1000]
        self.TrackInfo = 'track.xlsx'
        self.Offset = [10034, 7866]
        self.ChipRect = [100, 100, 1000, 1000]
        self.RegisterScore = 1.0


class TissueCutInfo(QcInfo):
    def __init__(self):
        self.OutScore: None
        self.AvgScore: None


class BGIStitch(QcInfo):
    def __init__(self):
        self.PreStitchColPair = None  # [[3, 2], [2, 2]]
        self.PreStitchRowPair = None  # [[2, 3], [2, 2]]
        self.RawOffset = None         # 'raw_offset.xlsx'
        self.ValueOffset = None       # 'value_offset.xlsx'
        self.GlobalHeight = None      # 20000
        self.GlobalWidth = None       # 30000
        self.GlobalLoc = None         # bgi_location.xlsx


class ScopeStitch(QcInfo):
    def __init__(self):
        self.GlobalHeight = None      # 20000
        self.GlobalWidth = None       # 30000
        self.GlobalLoc = None         # scope_location.xlsx
        self.ScopeOffset = None       # scope_offset.xlsx


class StitchEval(QcInfo):
    def __init__(self):
        self.Flag = True             # True or False
        self.Rotation = None          # -0.1637017817341799
        self.Scale = None             # 0.7619047619047619
        self.TemplateSrc = None       # [3, 5]
        self.CredibleOffset = None    # credible_offset.xlsx
        self.GlobalHeight = None      # 20000
        self.GlobalWidth = None       # 30000
        self.GlobalLoc = None         # eval_location.txt
        self.GlobalTemplate = None    # global_template.txt
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
        self.GPU = False
        self.Watershed = False


class ImageInfo(QcInfo):
    def __init__(self):
        self.WhiteBalance = None
        self.Illuminance = None
        self.Gain = None
        self.TempPath = ''
        self.ImagePath = ''
        self.Manufacturer = 'Motic'
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
        self.RGBScale = [1.0, 1.0, 1.0]
        self.Brightness = 50
        self.ColorEnhancement = 0
        self.Contrast = 0
        self.Gamma = 1.000000
        self.GammaShift = 0
        self.Sharpness = 0
        self.DistortionCorrection = False
        self.BackgroundBalance = False
        self.SlideInfo = 'FP2_20210527_SC_B3_4'
        self.Experimenter = 'wanglulu@genomics.cn'
        self.Contrast = 0


class Operation(QcInfo):
    def __init__(self):
        self.QCEval = False
        self.Stitching = True
        self.Register = True
        self.Segment = True
        self.Analysis = True


class UploadInfo(QcInfo):
    def __init__(self):
        self.MD5 = ''
        self.UploadTime = ''
        self.FileList = list()


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
        self.TissueCutInfo = TissueCutInfo()
        self.CellRecogInfo = CellRecogInfo()
        self.UploadInfo = UploadInfo()

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
            else:
                delattr(self, attr)


def main():
    si = ScopeInfo()
    si.serialize('E:\\Desktop\\scope_info.json')

    sj = ScopeInfo()
    sj.deserialize('E:\\Desktop\\scope_info.json')
    print(sj.ImageInfo.ImagePath)
    print(sj.ImageInfo.GUID)
    print(sj.ImageInfo.FOVHeight)


if __name__ == '__main__':
    main()
