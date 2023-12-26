import os
from os.path import split, join, dirname, abspath, exists
import glog
import numpy as np
import sys
sys.path.append(dirname(abspath(__file__)))
import common
import fov_json
import glob
# from fov_json import ScopeInfo
# from common import search_files, img_read, filename2index, location2npy


class ScopeStitchInfo(object):
    def __init__(self):
        self.global_height = None
        self.global_width = None
        self.loc = None
        self.overlap = None

    def is_stitched(self, ):
        return self.global_height is not None

    def from_json(self, output, scope_stitch):
        if scope_stitch.GlobalWidth is None:
            return
        self.global_height = scope_stitch.GlobalHeight
        self.global_width = scope_stitch.GlobalWidth
        # self.loc = common.location2npy(join(output, scope_stitch.GlobalLoc))


class FOVInfo(object):
    def __init__(self, json_path, imgs_path=None):
        if isinstance(json_path, str):
            self.output = split(json_path)[0]
            self.scope_info = fov_json.ScopeInfo()
            self.scope_info.deserialize(json_path)
        else:
            self.output = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), 'qcImgUpload')
            self.scope_info = json_path
        self.manufacturer = self.scope_info.ImageInfo.Manufacturer
        self.fov_width = self.scope_info.ImageInfo.FOVWidth
        self.fov_height = self.scope_info.ImageInfo.FOVHeight
        self.fov_cols = self.scope_info.ImageInfo.ScanCols
        self.fov_rows = self.scope_info.ImageInfo.ScanRows
        if not self.scope_info.ImageInfo.ScanRows:
            self.fov_deepth = 16
        else:
            self.fov_deepth = self.scope_info.ImageInfo.BitDepth
        self.fov_channel = 1
        if imgs_path is None:
            self.fov_image_path = self.scope_info.ImageInfo.TempPath
        else:
            self.fov_image_path = imgs_path
        self.stitch_info = self.scope_info.StitchInfo
        self.scope_stitch_info = ScopeStitchInfo()
        self.scope_stitch_info.from_json(self.output, self.stitch_info.ScopeStitch)
        self.standard_template = self.scope_info.ChipInfo.FOVTrackTemplate

        self.fov_count = 0
        self.bg_color = None
        self.fov_spatial_src = self.deploy_imgs()
        glog.info('{} {}-FOV arrangement in {} X {} grid, as shape is {} X {}.'.format(
            self.fov_count, self.manufacturer, self.fov_rows, self.fov_cols, self.fov_height, self.fov_width))

    def imgs_father_path(self, ):
        return abspath(dirname(self.fov_image_path))

    def set_output_path(self, file_path):
        if not exists(file_path):
            os.makedirs(file_path)
        self.output = file_path

    def is_stitched(self, ):
        return self.scope_stitch_info.is_stitched()

    def deploy_imgs(self, ):
        imgs_path = self.fov_image_path
        glog.info('Get imgs from {}.'.format(imgs_path))
        if self.manufacturer == "leica dm6b":
            imgs = glob.glob(os.path.join(imgs_path, 'TileScan*.tif'))
        else:
            imgs = common.search_files(imgs_path, ['.png', '.tif'])
        if self.manufacturer == 'leica':  # pass the image is stitched.
            merge = [x for i, x in enumerate(imgs) if x.find('Merging_Crop') != -1]
            imgs = [it for it in imgs if it not in merge]

        import cv2 as cv
        if self.manufacturer == 'czi':
            mat = cv.imread(imgs[0])
        else:
            mat = cv.imread(imgs[0], -1)
        self.fov_channel = common.mat_channel(mat)
        self.scope_info.ImageInfo.Channel = self.fov_channel
        self.fov_deepth = common.mat_deepth(mat)
        self.scope_info.ImageInfo.BitDepth = self.fov_deepth

        self.fov_count = len(imgs)
        spatial_src = np.empty((self.fov_rows, self.fov_cols), dtype='S256')
        flag = np.empty((self.fov_rows, self.fov_cols), dtype=bool)
        flag[::] = False
        for img in imgs:
            if self.manufacturer == 'leica':
                c, r = common.filename2index(img, self.manufacturer, row_len=self.fov_rows)
            elif self.manufacturer == 'leica dm6b':
                c, r = common.filename2index(img, self.manufacturer, row_len=self.fov_cols)
            else:
                c, r = common.filename2index(img, self.manufacturer)
            spatial_src[r, c] = img
            flag[r, c] = True
        ind = np.where(flag == False)
        spatial_src[ind] = None
        return spatial_src

    def no_none_row_ind(self, row_ind):
        for i in range(self.fov_cols):
            img = self.get_image_path(row_ind, i)
            if img != 'None':
                return i

    def no_none_col_ind(self, col_ind):
        for i in range(self.fov_rows):
            img = self.get_image_path(i, col_ind)
            if img != 'None':
                return i

    def get_image_path(self, fov_row_ind, fov_col_ind):
        return str(self.fov_spatial_src[fov_row_ind, fov_col_ind], encoding='utf-8')

    def get_image(self, fov_row_ind, fov_col_ind):
        image_path = self.get_image_path(fov_row_ind, fov_col_ind)
        img = common.img_read(image_path)
        return img

    def get_image3(self, fov_row_ind, fov_col_ind):
        image_path = self.get_image_path(fov_row_ind, fov_col_ind)
        # glog.info('{}-{}, {}.'.format(fov_row_ind, fov_col_ind, image_path))
        return common.img_read3(image_path)

