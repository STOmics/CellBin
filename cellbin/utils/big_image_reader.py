from .fov_reader import FOVReader
# from fov_reader import FOVReader
import os.path as osp
import glog
import os
import cv2 as cv
import math
import slideio
import tifffile as tifi
import numpy as np


class BigImageReader(FOVReader):
    def __init__(self, file_path, save_path, manufacturer, chip_name, overlap,
                 fov_height, fov_width, scope_info=None):
        super(BigImageReader, self).__init__(file_path, scope_info)
        self.manufacturer = manufacturer
        self.file_path = file_path
        self.save_path = save_path
        self.chip_name = chip_name
        self.overlap = overlap
        self.fov_height = fov_height
        self.fov_width = fov_width
        self.cut_shape = None
        if not osp.exists(self.file_path):
            glog.info(f"Input path -> {file_path} does not exist")
        else:
            self.cur_save_path = osp.join(self.save_path, f"{chip_name}")
            os.makedirs(self.cur_save_path, exist_ok=True)
            self.cut_image()
        glog.info(f"Image path -> {self.file_path}")

    def cut_image(self):
        try:
            image = tifi.imread(self.file_path)
        except:
            image = cv.imdecode(np.fromfile(self.file_path, dtype=np.uint8), -1)
        original_shape = image.shape
        if original_shape[0] > original_shape[-1]:
            image = image[..., 0]
        elif original_shape[0] < original_shape[-1]:
            image = image[0, ...]
        original_shape = image.shape
        cut_rate = 1 - self.overlap
        cut_size = [round(self.fov_height * cut_rate), round(self.fov_width * cut_rate)]
        col_num = math.ceil(original_shape[1] * 1.0 / cut_size[1])
        row_num = math.ceil(original_shape[0] * 1.0 / cut_size[0])
        self.cut_shape = row_num, col_num
        self.scope_info.ImageInfo.ScanCols = col_num
        self.scope_info.ImageInfo.ScanRows = row_num
        self.stitch_info.loc = np.zeros((self.scope_info.ImageInfo.ScanRows,
                                         self.scope_info.ImageInfo.ScanCols, 2), dtype=int)
        self.scope_info.ImageInfo.Overlap = self.overlap
        self.small_images = dict()
        for i in range(row_num):
            for j in range(col_num):
                start_x = j * cut_size[1]
                start_y = i * cut_size[0]
                end_x = min((j + 1) * cut_size[1], original_shape[1])
                end_y = min((i + 1) * cut_size[0], original_shape[0])
                cur_image = image[start_y: end_y, start_x: end_x]
                self.small_images[str(i)+'_'+str(j)] = cur_image
                self.stitch_info.loc[i, j] = [start_x, start_y]
                tif_name = "cut_{}_{:0>4d}_{:0>4d}.tif".format(self.chip_name, i, j)
                img_save_path = os.path.join(self.cur_save_path, tif_name)
                cv.imwrite(img_save_path, cur_image)
        self.scope_info.StitchInfo.ScopeStitch.GlobalHeight = original_shape[0]
        self.scope_info.StitchInfo.ScopeStitch.GlobalWidth = original_shape[1]
        self.scope_info.ImageInfo.ImagePath = self.file_path
        self.scope_info.ImageInfo.TempPath = self.cur_save_path
        self.scope_info.ImageInfo.Manufacturer = self.manufacturer
        self.scope_info.ImageInfo.FOVWidth = self.fov_width
        self.scope_info.ImageInfo.FOVHeight = self.fov_height
        self.scope_info.ImageInfo.stitched_image = True
        glog.info(f"Finished cutting")

    def get_cut_shape(self):
        return self.cut_shape


if __name__ == '__main__':
    img_path = r"D:\Data\brain-GE-merge_RAW_ch00.tif"
    save_path = r"D:\Data\test"
    chip_name = "ss2"
    manufacturer = "Leica"
    overlap = 0.1
    fov_height = 2040
    fov_width = 2040
    br = BigImageReader(img_path, save_path, manufacturer, chip_name, overlap,
                 fov_height, fov_width)
    br.generate_xlsx(save_path, "ss2")