import glog
import numpy as np
from os.path import join, basename, dirname, abspath
import sys
sys.path.append(join(dirname(abspath(__file__)), '../..'))
from utils import common, fov_json


class StitchInfo(object):
    def __init__(self):
        self.global_height = None
        self.global_width = None
        self.loc = None
        self.overlap = None

    def is_stitched(self, ):
        return self.loc is not None


class FOVReader(object):
    def __init__(self, file_path, scope_info=None):
        self.manufacturer = ''  # zeiss, motic
        self.file_path = file_path
        self.fov_width = None
        self.fov_height = None
        self.fov_cols = None
        self.fov_rows = None
        self.fov_count = 0
        self.bg_color = None
        self.stitch_info = StitchInfo()
        if not scope_info:
            self.scope_info = fov_json.ScopeInfo()
        else:
            self.scope_info = scope_info

    def is_stitched(self, ):
        return self.stitch_info.is_stitched()

    def set_chip_info(self, chip_no, info):
        self.scope_info.ChipInfo.ChipID = chip_no
        self.scope_info.ChipInfo.Pitch = info['chipPitch']
        self.scope_info.ChipInfo.FOVTrackTemplate = [info['trackPatternBlockSizeX'], info['trackPatternBlockSizeY']]

    def set_scope_info_from_imgs(self, imgs_path):
        imgs = common.search_files(imgs_path, ['.png', '.tif'])
        if self.manufacturer == 'leica':  # pass the image is stitched.
            merge = [x for i, x in enumerate(imgs) if x.find('Merging_Crop') != -1]
            imgs = [it for it in imgs if it not in merge]
            ind = [common.filename2index(basename(i), self.manufacturer, row_len=self.fov_rows) for i in imgs]
        else:
            ind = [common.filename2index(basename(i), self.manufacturer) for i in imgs]
        ind = np.array(ind)
        self.fov_cols = int(np.max(ind[:, 0]) + 1)
        self.fov_rows = int(np.max(ind[:, 1]) + 1)
        self.fov_count = self.fov_cols * self.fov_rows
        arr = common.img_read(imgs[0])
        self.fov_height, self.fov_width = arr.shape

        self.scope_info.ImageInfo.ScanRows = self.fov_rows
        self.scope_info.ImageInfo.ScanCols = self.fov_cols
        self.scope_info.ImageInfo.FOVHeight = self.fov_height
        self.scope_info.ImageInfo.FOVWidth = self.fov_width
        self.scope_info.ImageInfo.TempPath = imgs_path
        self.scope_info.ImageInfo.Manufacturer = self.manufacturer
        self.scope_info.ImageInfo.ImagePath = ''

    def generate_json(self, file_path):
        if self.stitch_info.is_stitched():
            dct = {'location': self.stitch_info.loc}
            self.scope_info.StitchInfo.ScopeStitch.GlobalLoc = '{}_{}_stitch.xlsx'.format(
                self.scope_info.ChipInfo.ChipID, self.manufacturer)
            xlsx_path = join(file_path, self.scope_info.StitchInfo.ScopeStitch.GlobalLoc)
            common.npy2xlsx(dct, xlsx_path)
            glog.info('Save scope stitching info to {}.'.format(xlsx_path))
        json_path = join(file_path, 'bgi_{}.json'.format(self.scope_info.ChipInfo.ChipID))
        self.scope_info.serialize(json_path)
        glog.info('Save json file to {}.'.format(json_path))

    def generate_xlsx(self, out_path, out_name):
        if self.stitch_info.is_stitched():
            dct = {'location': self.stitch_info.loc}
            self.scope_info.StitchInfo.ScopeStitch.GlobalLoc = out_name + '.xlsx'
            xlsx_path = join(out_path, self.scope_info.StitchInfo.ScopeStitch.GlobalLoc)
            common.npy2xlsx(dct, xlsx_path)
            glog.info('Save scope stitching info to {}.'.format(xlsx_path))
            return xlsx_path


def test():
    glog.info('Next test the performance of Reader.')
    # file_path = 'D:\\DATA\\stitching\\motic\\20210527_SC_A3_2'
    # frf = FOVReaderFactory()
    # read = frf.create_reader('motic', file_path)
    # print(read.__dict__)
    # print(read.stitch_info.__dict__)
    # print(read.is_stitched())

    # file_path = 'D:\\DATA\zeiss\\raw\\20210426-T167-Z2-L-M019-01.czi'
    # file_path = 'D:\\DATA\zeiss\\20210426-T167-Z2-L-M019-01'
    # frf = FOVReaderFactory()
    # reader = frf.create_reader('zeiss', file_path)
    # print(reader.__dict__)
    # print(reader.stitch_info.__dict__)
    # print(reader.is_stitched())


def main():
    test()


if __name__ == '__main__':
    main()
