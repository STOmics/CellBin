import os
import time
from skimage import measure
from os.path import join, splitext, exists, split
import tifffile
import cv2
import numpy as np

import cellbin.cell_segmentation.seg_utils.grade as grade
import cellbin.cell_segmentation.seg_utils.utils as utils
import cellbin.cell_segmentation.seg_utils.transform as transform
import torch
import glog


class CellSegPipe(object):
    """ Pipeline of cell segmentation """

    def __init__(self, img_path, out_path, is_water=0, DEEP_CROP_SIZE=20000, OVERLAP=100):
        """Initialize segmentation

        Args:
            img_path: path of stain image
            out_path: path of the cell mask
            is_water: if use watershed method or not
            DEEP_CROP_SIZE:
            OVERLAP: When splitting, the overlap rate between adjacent graphs
        """
        self.deep_crop_size = DEEP_CROP_SIZE
        self.overlap = OVERLAP
        self.__img_path = img_path
        if os.path.isdir(img_path):
            self.__file = os.listdir(img_path)
            self.__is_list = True
        else:
            self.__file = [split(img_path)[-1]]
            self.__is_list = False
        self.__file_name = [splitext(file)[0] for file in self.__file]
        self.__img_suffix = [splitext(file)[-1] for file in self.__file]
        self.img_list = self.__imload_list(img_path)
        self.__convert_gray()
        self.__out_path = out_path
        output_dir = os.path.dirname(self.__out_path)
        if not exists(output_dir):
            os.mkdir(output_dir)
            glog.info('Create new dir : %s' % output_dir)
        self.__is_water = is_water
        t0 = time.time()
        self.__trans16to8()
        t1 = time.time()
        glog.info('Transform 16bit to 8bit : %.2f' % (t1 - t0))
        self.tissue_mask = []
        self.tissue_mask_thumb = []
        self.tissue_num = []  # tissue num in each image
        self.tissue_bbox = []  # tissue roi bbox in each image
        # self.img_filter = []  # image filtered by tissue mask
        self.cell_mask = []
        self.post_mask_list = []
        self.score_mask_list = []
        self.tissue_cell_label = None
        self.cell_label = None
        self.image_preprocess = []

    def __imload_list(self, img_path):
        """
        Args:
            img_path: path of input image
        """
        if self.__is_list:
            img_list = []
            for idx, file in enumerate(self.__file):
                img_temp = self.__imload(join(img_path, file), idx)
                img_list.append(img_temp)
            return img_list
        else:
            img_temp = self.__imload(img_path, 0)
            return [img_temp]

    def __imload(self, img_path, id):

        assert self.__img_suffix[id] in ['.tif', '.png', '.jpg']
        if self.__img_suffix[id] == '.tif':
            img = tifffile.imread(img_path)
        else:
            img = cv2.imread(img_path, -1)
        return img

    def __convert_gray(self):

        for idx, img in enumerate(self.img_list):
            if len(img.shape) == 3:
                glog.info('Image %s convert to gray!' % self.__file[idx])
                self.img_list[idx] = img[:, :, 0]

    def __trans16to8(self):

        for idx, img in enumerate(self.img_list):
            assert img.dtype in ['uint16', 'uint8']
            if img.dtype != 'uint8':
                glog.info('%s transfer to 8bit' % self.__file[idx])
                self.img_list[idx] = utils.transfer_16bit_to_8bit(img)

    def __get_img_filter(self):

        """get tissue image by tissue mask"""
        for idx, img in enumerate(self.img_list):
            img_filter = np.multiply(img, self.tissue_mask[idx]).astype(np.uint8)
            self.img_filter.append(img_filter)

    def __filter_roi(self, props):

        filtered_props = []
        for id, p in enumerate(props):

            black = np.sum(p['intensity_image'] == 0)
            sum = p['bbox_area']
            ratio_black = black / sum
            pixel_light_sum = np.sum(np.unique(p['intensity_image']) > 128)
            if ratio_black < 0.75 and pixel_light_sum > 10:
                filtered_props.append(p)
        return filtered_props

    def __get_roi(self):

        """get tissue area from ssdna"""
        for idx, tissue_mask in enumerate(self.tissue_mask):

            label_image = measure.label(tissue_mask, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=self.img_list[idx])

            # remove noise tissue mask
            filtered_props = props  # self.__filter_roi(props)
            if len(props) != len(filtered_props):
                tissue_mask_filter = np.zeros((tissue_mask.shape), dtype=np.uint8)
                for tissue_tile in filtered_props:
                    bbox = tissue_tile['bbox']
                    tissue_mask_filter[bbox[0]: bbox[2], bbox[1]: bbox[3]] += tissue_tile['image']
                self.tissue_mask[idx] = np.uint8(tissue_mask_filter > 0)
                self.tissue_mask_thumb[idx] = tissue_seg.down_sample(self.tissue_mask[idx])
            self.tissue_num.append(len(filtered_props))
            self.tissue_bbox.append([p['bbox'] for p in filtered_props])

    def preprocess(self):
        for idx, img in enumerate(self.img_list):
            trsf_img = transform.transform(img)
            # if torch.cuda.is_available():
            #     self.image_preprocess.append(trsf_img.denoise_with_median_cuda())
            # else:
            #     self.image_preprocess.append(trsf_img.denoise_with_median(processes=20))
            tmp_img = trsf_img.mesmer_prepro()

            # trsf_img1 = transform.transform(tmp_img)
            # tmp_img1 = trsf_img1.mesmer_prepro()
            # tifffile.imsave(
            #     '/jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/liaoxiangxiang/tmp/tmp_result/DAPI_fov_stitched_transformed_register_preproc.tif',
            #     tmp_img1)
            self.image_preprocess.append(tmp_img)

    def tissue_cell_infer(self, batch_size=20):
        import seg_utils.cell_infer as cell_infer

        """cell segmentation in tissue area by neural network"""
        self.tissue_cell_label = []
        for idx, img in enumerate(self.image_preprocess):
            tissue_bbox = self.tissue_bbox[idx]
            tissue_img = [img[p[0]: p[2], p[1]: p[3]] for p in tissue_bbox]
            cell_seg = cell_infer.CellInfer(tissue_img, batch_size)
            label_list = cell_seg.run_infer()
            self.tissue_cell_label.append(label_list)
        return 0
        # q.put(0)
        # q.put(tissue_cell_label)

    def cell_infer(self, batch_size=20):
        import seg_utils.cell_infer as cell_infer

        """cell segmentation by neural network"""
        self.cell_label = []
        for idx, img in enumerate(self.img_list):
            cell_seg = cell_infer.CellInfer(img, batch_size)
            label_list = cell_seg.run_infer()
            self.cell_label.append(label_list)
        return 0

    def tissue_label_filter(self, tissue_cell_label):

        """filter cell mask in tissue area"""
        tissue_cell_label_filter = []
        for idx, label in enumerate(tissue_cell_label):
            tissue_bbox = self.tissue_bbox[idx]
            label_filter_list = []
            for i in range(self.tissue_num[idx]):
                tissue_bbox_temp = tissue_bbox[i]
                label_filter = np.multiply(label[i], self.tissue_mask[idx][tissue_bbox_temp[0]: tissue_bbox_temp[2],
                                                     tissue_bbox_temp[1]: tissue_bbox_temp[3]]).astype(np.uint8)
                label_filter_list.append(label_filter)
            tissue_cell_label_filter.append(label_filter_list)
        return tissue_cell_label_filter

    def __mosaic(self, tissue_cell_label_filter):

        """mosaic tissue into original mask"""
        for idx, label_list in enumerate(tissue_cell_label_filter):
            tissue_bbox = self.tissue_bbox[idx]
            cell_mask = np.zeros((self.img_list[idx].shape), dtype=np.uint8)
            for i in range(self.tissue_num[idx]):
                tissue_bbox_temp = tissue_bbox[i]
                cell_mask[tissue_bbox_temp[0]: tissue_bbox_temp[2],
                tissue_bbox_temp[1]: tissue_bbox_temp[3]] = label_list[i]
            self.cell_mask.append(cell_mask)
        return self.cell_mask

    def watershed_score(self, cell_mask):
        """watershed and score on cell mask by neural network"""

        for idx, cell_mask in enumerate(cell_mask):
            cell_mask = np.squeeze(np.asarray(cell_mask))

            cell_mask = grade.edgeSmooth(cell_mask)
            cell_mask_tile, x_list, y_list = utils.split(cell_mask, self.deep_crop_size)
            img_tile, _, _ = utils.split(self.img_list[idx], self.deep_crop_size)
            input_list = [[cell_mask_tile[id], img] for id, img in enumerate(img_tile)]
            if self.__is_water:
                post_list_tile = grade.watershed_multi(input_list, 20)
            else:
                post_list_tile = grade.score_multi(input_list, 20)

            post_mask_tile = [label[0] for label in post_list_tile]
            score_mask_tile = [label[1] for label in post_list_tile]  # grade saved
            post_mask = utils.merge(post_mask_tile, x_list, y_list, cell_mask.shape)
            score_mask = utils.merge(score_mask_tile, x_list, y_list, cell_mask.shape)

            # post_mask = grade.edgeSmooth(post_mask)

            self.post_mask_list.append(post_mask)
            self.score_mask_list.append(score_mask)
            # post_mask = grade.edgeSmooth(cell_mask)
        # self.post_mask_list.append(post_mask)

    def post_process(self, cell_masks):
        """post_process on cell mask by neural network"""

        for idx, cell_mask in enumerate(cell_masks):
            cell_mask = np.squeeze(np.asarray(cell_mask))
            post_mask = grade.edgeSmooth(cell_mask)
            if np.max(post_mask) == 1: post_mask = post_mask * 255
            self.post_mask_list.append(post_mask)

    def save_tissue_mask(self):

        for idx, tissue_thumb in enumerate(self.tissue_mask_thumb):
            tifffile.imsave(join(self.__out_path, self.__file_name[idx] + r'_tissue_cut.tif'), tissue_thumb)

    def __mkdir_subpkg(self):
        """make new dir while image is large"""

        assert self.__is_list == False
        file = self.__file[0]
        file_name = splitext(file)[0]

        mask_outline_name = r'_watershed_outline' if self.__is_water else r'_outline'
        mask_name = r'_watershed' if self.__is_water else r''

        self.__subpkg_mask = join(self.__out_path, file_name + mask_name)
        self.__subpkg_mask_outline = join(self.__out_path, file_name + mask_outline_name)
        self.__subpkg_score = join(self.__out_path, file_name + r'_score')

        if not os.path.exists(self.__subpkg_mask):
            os.mkdir(self.__subpkg_mask)

        if not os.path.exists(self.__subpkg_mask_outline):
            os.mkdir(self.__subpkg_mask_outline)

        if not os.path.exists(self.__subpkg_score):
            os.mkdir(self.__subpkg_score)

    def __save_each_file_result(self, file_name, idx):
        mask_name = r'nuclei_mask.tif'
        tifffile.imwrite(join(self.__out_path, file_name + mask_name),
                         self.post_mask_list[idx], compression="zlib", compressionargs={"level": 8})

    def save_cell_mask(self):
        """save cell mask from network or watershed"""
        if len(self.__file) == 1:
            if np.max(self.post_mask_list[0]) == 1: self.post_mask_list[0] *= 255
            tifffile.imwrite(self.__out_path, self.post_mask_list[0], compression="zlib", compressionargs={"level": 8})

    def save_result(self):
        """save tissue mask"""

        self.save_cell_mask()

    def run_tissue_cell_seg(self):
        """ Inference for tissue Segmentation """
        t0 = time.time()
        try:
            self.tissue_mask = [tifffile.imread(os.path.join(self.__out_path, self.__file_name[0] + '_tissue_cut.tif'))]
        except:
            self.tissue_mask = [np.ones((self.img_list[0].shape), dtype=np.uint8)]
        self.tissue_mask_thumb = [utils.down_sample(mask) for mask in self.tissue_mask]

        self.__get_img_filter()
        self.__get_roi()
        glog.info('Start do cell mask, this will take some minutes.')
        t1 = time.time()

        self.tissue_cell_infer()
        t2 = time.time()
        glog.info('Cell inference : %.2f' % (t2 - t1))

        ###filter by tissue mask###
        tissue_cell_label_filter = self.tissue_label_filter(self.tissue_cell_label)
        t3 = time.time()
        glog.info('Filter by tissue mask : %.2f' % (t3 - t2))

        ###mosaic tissue roi ###
        cell_mask = self.__mosaic(tissue_cell_label_filter)
        t4 = time.time()
        glog.info('Mosaic tissue roi : %.2f' % (t4 - t3))

        ###post process###
        self.watershed_score(cell_mask)
        t5 = time.time()
        glog.info('Post-processing : %.2f' % (t5 - t4))

        self.save_result()
        glog.info('Result saved : %s ' % (self.__out_path))

    def run_cell_seg(self):
        """ Inference for cell Segmentation """
        torch.set_num_threads(20)
        ### preprocess ###
        t1 = time.time()
        glog.info('start do preprocess ')
        self.preprocess()
        t2 = time.time()
        glog.info('preprocess time : %.2f' % (t2 - t1))

        ### cell segmentation ###
        glog.info('Start do cell mask, this will take some minutes.')
        self.cell_infer(batch_size=20)
        t3 = time.time()
        glog.info('Cell inference time: %.2f' % (t3 - t2))

        ### post process ###
        glog.info('start do postprocess and watershed')
        # self.post_process(self.cell_label)
        self.watershed_score(self.cell_label)
        t4 = time.time()
        glog.info('postprocess time : %.2f' % (t4 - t3))

        ### save result ###
        self.save_result()
