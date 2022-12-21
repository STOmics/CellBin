# Copyright 2022 Beijing Genomics Institute(BGI) Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2 as cv
from stitch.dft_util import argmax_translation, _apodize
from skimage.metrics import structural_similarity as ssim
import itertools


class Matcher(object):
    """ Matcher provide the way to get overlap/offset from neighbor FOV. """
    def __init__(self, ):
        self.overlap = 0.1
        self.horizontal = True

    def get_roi(self, arr0, arr1):
        # if shape is not equal?
        h0, w0 = arr0.shape
        h1, w1 = arr1.shape
        w = int(min(w0, w1) * self.overlap)
        h = int(min(h0, h1) * self.overlap)

        if self.horizontal:
            x0, y0, x1, y1 = [w0 - w, 0, w0, h0]
            x0_, y0_, x1_, y1_ = [0, 0, w, h1]
        else:
            x0, y0, x1, y1 = [0, h0 - h, w0, h0]
            x0_, y0_, x1_, y1_ = [0, 0, w1, h]
        return arr0[y0: y1, x0: x1], arr1[y0_: y1_, x0_: x1_]

    @staticmethod
    def slice_image(image: np.ndarray, slice_width: int, slice_height: int,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """
        Image is cropped and stacked
        """
        if image.ndim == 3:
            image_height, image_width, _ = image.shape
        else: image_height, image_width = image.shape

        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)

        slice_bboxes = []
        y_max = y_min = 0
        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap

        n_ims = 0
        sliced_image_result = []
        assert len(slice_bboxes) > 0, "slice image must > 0"
        for slice_bbox in slice_bboxes:
            n_ims += 1
            # extract image
            tlx = slice_bbox[0]
            tly = slice_bbox[1]
            brx = slice_bbox[2]
            bry = slice_bbox[3]
            image_pil_slice = image[tly:bry, tlx:brx]
            sliced_image_result.append(image_pil_slice)
        return sliced_image_result


class SIFTMatcher(Matcher):
    def __init__(self, ):
        super(SIFTMatcher, self).__init__()
        self.sift = cv.SIFT_create()
        self.__matcher = self.matcher()

    def neighbor_match(self, train, query):
        train_local, query_local = self.get_roi(train, query)
        train_local = np.array(train_local, dtype=np.uint8)
        query_local = np.array(query_local, dtype=np.uint8)
        # train_local = cv.equalizeHist(train_local)
        # query_local = cv.equalizeHist(query_local)
        return self.sift_match(train_local, query_local)

    def sift_match(self, train, query):
        psd_kp1, psd_des1 = self.sift.detectAndCompute(train, None)
        kps1 = np.float32([kp.pt for kp in psd_kp1])
        psd_kp2, psd_des2 = self.sift.detectAndCompute(query, None)
        kps2 = np.float32([kp.pt for kp in psd_kp2])
        if len(kps1) == 0 or len(kps2) == 0:
            return None
        try:
            matches = self.__matcher.knnMatch(psd_des1, psd_des2, k=2)
        except:
            return None
        local_match = list()

        for m, n in matches:
            if m.distance < 0.5 * n.distance: local_match.append((m.trainIdx, n.queryIdx))

        if len(local_match) == 0: return None
        else:
            pts_a = np.float32([kps1[i] for (_, i) in local_match])
            pts_b = np.float32([kps2[i] for (i, _) in local_match])
            if self.horizontal: pts_b[:, 0] += train.shape[1]
            else: pts_b[:, 1] += train.shape[0]
            offset = np.median(pts_a - pts_b, axis=0)
            return [offset[0], offset[1], 100]

    @staticmethod
    def matcher():
        flann_index_kd_tree = 1
        index_params = dict(algorithm=flann_index_kd_tree, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        return flann


class FFTMatcher(Matcher):
    def __init__(self):
        super(FFTMatcher, self).__init__()

    @staticmethod
    def hanning_windows(matrix, win_size=5):
        '''
        hanning windows in image
        :param src:
        :param dst:
        :param filter_size:
        :return:
        '''
        # matrix1 = np.transpose(matrix,(1,2,0))
        # Handle relatively large values in the upper, lower, left, and right corners, the size of win_size
        filter_value = matrix.mean()
        filter_size = int(np.ceil(win_size / 2))
        matrix[:filter_size, :filter_size] = filter_value
        matrix[-filter_size:, :filter_size] = filter_value
        matrix[:filter_size, -filter_size:] = filter_value
        matrix[-filter_size:, -filter_size:] = filter_value
        if win_size is int:
            win = cv.createHanningWindow((win_size, win_size), cv.CV_32F)
            matrix = cv.filter2D(matrix, -1, win)
        return matrix

    @staticmethod
    def gaussian_blur(src, dst, win_size=5):
        src = np.transpose(src, (1, 2, 0))
        src = cv.GaussianBlur(src, (win_size, win_size), 0)
        src = np.transpose(src, (2, 0, 1))
        dst = np.transpose(dst, (1, 2, 0))
        dst = cv.GaussianBlur(dst, (win_size, win_size), 0)
        dst = np.transpose(dst, (2, 0, 1))
        return src, dst

    def dirac_func_n(self, src, dst, win=False):
        '''
        dirac function to calculate peak offset in the fft of cross-power specturm
        :param src: [AxNxM], image of source, A is number of image to calculate cross-power specturm
        :param dst: [AxNxM], image of destination
        :return: offset
        '''
        assert np.ndim(dst) == 3, "inputting image must be gray"
        # src,dst = self.gaussian_blur(src,dst)
        src_fft = np.fft.fft2(src)
        dst_fft = np.fft.fft2(dst)
        dirac_result = np.fft.ifft2(np.multiply(dst_fft, src_fft.conj()) /
                                    (np.abs(np.multiply(dst_fft, src_fft.conj())) + np.finfo(np.float32).eps))
        dirac_result = dirac_result.real

        filter_value = 0
        filter_size = 3
        dirac_result[:, :filter_size, :filter_size] = filter_value
        dirac_result[:, -filter_size:, :filter_size] = filter_value
        dirac_result[:, :filter_size, -filter_size:] = filter_value
        dirac_result[:, -filter_size:, -filter_size:] = filter_value
        sum_result = np.zeros(dirac_result[0].shape)

        for dirac in dirac_result:
            if np.abs(dirac.max()) < 100:
                if win:
                    dirac = self.hanning_windows(dirac, win_size=7)
                sum_result += dirac  # *(dirac.max()/sum_enger_all)
        offset = np.where(sum_result == np.amax(sum_result))  # [row,col]-->[y,x]
        y, x = offset[0][0], offset[1][0]
        confidence = sum_result.max() - (sum_result.mean() + 5 * sum_result.std())
        return [x, y, confidence]

    def statck_imgs(self, src, dst):
        """
        add up the stacked images
        :param src: CxMXN
        :param dst:CxMXN
        :return:
        """
        src, dst = np.array(src), np.array(dst)
        src = np.expand_dims(np.sum(src, axis=0), axis=0)
        dst = np.expand_dims(np.sum(dst, axis=0), axis=0)
        return src, dst

    def slice_images(self, src_img, dst_img, direction):
        """
        crop and flod images
        :param src_img: template image
        :param dst_img: background image
        :param direction: 0:right, 1:down, 2: left, 3: up, which is src->dst
        :return: flod images
        """
        overlap_size = min(dst_img.shape[:2])
        ratio = 0.7
        width = overlap_size if direction % 2 == 0 else int(overlap_size*ratio)
        height = int(overlap_size*ratio) if direction % 2 == 0 else overlap_size
        src_slice_img = self.slice_image(src_img, slice_width=width, slice_height=height,
                                    overlap_height_ratio=0.2, overlap_width_ratio=0.2, )
        # src_slice_img = src_slice_img.images
        dst_slice_img = self.slice_image(dst_img, slice_width=width, slice_height=height,
                                    overlap_height_ratio=0.2, overlap_width_ratio=0.2, )
        # dst_slice_img = dst_slice_img.images
        src_slice_img, dst_slice_img = self.fliter_null_information_image(src_slice_img, dst_slice_img)
        return np.array(src_slice_img), np.array(dst_slice_img)

    @staticmethod
    def fliter_null_information_image(src_imgs, dst_imgs):
        """
        Filter out ROI with no information in a clipping area
        :param src_imgs:
        :param dst_imgs:
        :return:
        """
        src_lists = []
        dst_lists = []

        for src, dst in zip(src_imgs, dst_imgs):
            src_grad = cv.Sobel(src, -1, 1, 1).std()
            dst_grad = cv.Sobel(dst, -1, 1, 1).std()
            if max(src_grad, dst_grad) > 0.2:
                # src,dst = self.window_and_low_pass(src,dst)
                src_lists.append(src)
                dst_lists.append(dst)
            else:
                # print("src std: {}, dst std: {}".format(src_grad, dst_grad))
                pass
        return src_lists, dst_lists

    def neighbor_match_v1(self, dst, src, win=False):
        '''
        Calculate the relative offsets of adjacent FOVs
        :param src: [NxM], image of source, A is number of image to calculate cross-power specturm
        :param dst:[NxM], image of destination
        :param direction: direction: 0:right, 1:down, 2: left, 3: up, which is src->dst
        :return: offset
        '''
        direction = 2 if self.horizontal else 3
        dst, src = self.get_roi(dst, src)
        src_images, dst_images = self.slice_images(src, dst, direction)
        if len(src_images) == 0:
            return None
        # else:
        #     return None
        min_h, min_w = dst_images[0].shape
        # src_stack,dst_stack = self.statck_imgs(src_images,dst_images)
        # src_images = np.concatenate([src_images,src_stack],axis=0)
        # dst_images = np.concatenate([dst_images,dst_stack],axis=0)
        x, y, confidence = self.dirac_func_n(src_images, dst_images, win)
        # (x,y) Indicates the position coordinates of src moving to dst
        if (x is None) or (confidence < 0.08):
            return None

        if direction == 2:
            x -= min_w
            if y > min_h * 0.8: y -= min_h
            offset = [x, y]
        elif direction == 3:
            y -= min_h
            if x > min_w * 0.8: x -= min_w
            offset = [x, y]
        else:
            assert direction is [2, 3], "direction is error"
        offset.append(confidence)
        return offset

    def neighbor_match_v2(self, src, dst):
        '''
        Calculate the relative offsets of adjacent FOVs
        :param src: [NxM], image of source, A is number of image to calculate cross-power specturm
        :param dst:[NxM], image of destination
        :param direction: direction: 0:right, 1:down, 2: left, 3: up, which is src->dst
        :return: offset
        '''
        direction = 2 if self.horizontal else 3
        src, dst = self.get_roi(src, dst)
        src_images, dst_images = self.slice_images(src, dst, direction)
        # src_images,dst_images = [src],[dst]
        if len(src_images) == 0:
            return None
        min_h, min_w = dst_images[0].shape
        (y0, x0), confidence0, _ = self._translation(dst_images, src_images)  # x0,y0 indicate offset of the SRC

        offset_x, offset_y = None,None
        if direction == 2:  # left
            # x0 (Positive number) means src moves to the right, x0 must be less than 0
            offset_x = x0 if x0 < 0 else -min_w + x0
            offset_y = -y0

            offset_x = -offset_x - min_w
            if offset_y > min_h * 0.8: offset_y -= min_h

        elif direction == 3:  # up
            offset_y = y0 if y0 < 0 else -min_h + y0
            offset_x = -x0

            offset_y = -offset_y - min_h
            if offset_x > min_w * 0.8: offset_x -= min_w
        else:
            assert direction is [2, 3], "direction is error"

        ssim_vaule = ssim(src, dst)
        return [offset_x, offset_y, confidence0 * 100, ssim_vaule*100]

    def neighbor_match(self, src, dst):
        """
        Calculate the relative offsets of adjacent FOVs
        :param src: [NxM], image of source, A is number of image to calculate cross-power specturm
        :param dst:[NxM], image of destination
        :param direction: direction: 0:right, 1:down, 2: left, 3: up, which is src->dst
        :return: offset
        """
        direction = 2 if self.horizontal else 3
        src, dst = self.get_roi(src, dst)
        min_h, min_w = dst.shape
        pcm = self.pcm(src, dst).real
        # tvec, success=argmax_translation(pcm,filter_pcorr=0, constraints=None, reports=None)
        yins, xins, pcm_ = self.multi_peak_max(pcm)
        size_y, size_x = dst.shape[:2]
        lims = np.array([[-size_y, size_y], [-size_x, size_x]])
        max_peak = self.interpret_translation(src, dst, yins, xins, *lims[0], *lims[1])
        ncc, offset_y, offset_x, sub_dst, sub_src = max_peak
        confidence0 = pcm_.max()
        # Local
        if sub_dst is not None:
            sub_dst_crop,sub_src_crop = self.slice_images(sub_dst, sub_src, direction)
        else:
            return None
        if len(sub_src_crop) > 0:
            (y0, x0), confidence0, fft_image = self._translation(sub_dst_crop, sub_src_crop)
            offset_x += x0
            offset_y += y0

        if direction == 2:
            offset_x = offset_x - min_w
        elif direction == 3:
            offset_y = offset_y - min_h
        else:
            assert direction is [2, 3], "direction is error"
        # ssim_vaule = ssim(sub_dst_crop,sub_src_crop)

        return [offset_x, offset_y, confidence0 * 100,ncc*100]

    def _translation(self, im0, im1, filter_pcorr=0, constraints=None, reports=None):
        """
        The plain wrapper for translation phase correlation, no big deal.
        """
        # Apodization and pcorr don't play along
        # im0, im1 = [utils._apodize(im, ratio=1) for im in (im0, im1)]
        ret, succ, image = self._phase_correlation(
            im0, im1,
            argmax_translation, filter_pcorr, constraints, reports)
        return ret, succ, image

    def _phase_correlation(self, im0, im1, callback=None, *args):
        """
        Computes phase correlation between im0 and im1

        Args:
            im0
            im1
            callback (function): Process the cross-power spectrum (i.e. choose
                coordinates of the best element, usually of the highest one).
                Defaults to :func:`imreg_dft.utils.argmax2D`

        Returns:
            tuple: The translation vector (Y, X). Translation vector of (0, 0)
                means that the two images match.
        """
        if callback is None:
            callback = argmax_translation

        # TODO: Implement some form of high-pass filtering of PHASE correlation
        im0 = [_apodize(im) for im in im0]
        im1 = [_apodize(im) for im in im1]
        f0, f1 = [np.fft.fft2(arr) for arr in (im0,im1)]
        # spectrum can be filtered (already),
        # so we have to take precaution against dividing by 0
        eps = abs(f1).max() * 1e-15
        # cps == cross-power spectrum of im0 and im1
        cps = abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
        # scps = shifted cps
        scps = np.fft.fftshift(cps)
        scps_result = np.zeros_like(scps[0])
        for i, scp in enumerate(scps):
            scp = cv.GaussianBlur(scp, (7, 7), 1)
            thread = np.percentile(scp, 95)
            scp[np.where(scp < thread)] = 0
            (_, _), success = callback(scp, *args)
            scps_result += scp * success
            # print("{}".format(i))
        (t0, t1), success = callback(scps_result, *args)  # (y,x)
        if success < 0.05:
            scps_result=cv.GaussianBlur(scps_result, (7, 7), 1)
            win = cv.createHanningWindow((7, 7), cv.CV_32F)
            scps_result = cv.filter2D(scps_result, -1, win)
            (t0, t1), _ = callback(scps_result, *args)  # (y,x)

        ret = np.array((np.round(t0), np.round(t1)))

        # _compensate_fftshift is not appropriate here, this is OK.
        #shape=[c,h,w]
        t0 -= scps[0].shape[0] // 2
        t1 -= scps[0].shape[1] // 2

        ret -= np.array(scps[0].shape, int) // 2
        return ret, success,scps_result

    def interpret_translation(self,
                              image1: np.ndarray,
                              image2: np.ndarray,
                              yins: np.ndarray,
                              xins: np.ndarray,
                              ymin: int,
                              ymax: int,
                              xmin: int,
                              xmax: int,
                              n: int = 2,
                              ):
        """Interpret the translation to find the translation with heighest ncc.

        Parameters
        ---------
        image1 : np.ndarray
            the first image (the dimension must be 2)
        image2 : np.ndarray
            the second image (the dimension must be 2)
        yins : IntArray
            the y positions estimated by PCM
        xins : IntArray
            the x positions estimated by PCM
        ymin : Int
            the minimum value of y (second last dim.)
        ymax : Int
            the maximum value of y (second last dim.)
        xmin : Int
            the minimum value of x (last dim.)
        xmax : Int
            the maximum value of x (last dim.)
        n : Int
            the number of peaks to check, default is 2.

        Returns
        -------
        _ncc : Float
            the highest ncc
        x : Int
            the selected x position
        y : Int
            the selected y position
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        sizeY = image1.shape[0]
        sizeX = image1.shape[1]
        assert np.all(0 <= yins) and np.all(yins < sizeY)
        assert np.all(0 <= xins) and np.all(xins < sizeX)

        _ncc = -np.infty
        y = 0
        x = 0
        overlap_img_1 = None
        overlap_img_2 = None

        ymagss = [yins, sizeY - yins]
        ymagss[1][ymagss[0] == 0] = 0
        xmagss = [xins, sizeX - xins]
        xmagss[1][xmagss[0] == 0] = 0

        # concatenate all the candidates
        _poss = []
        for ymags, xmags, ysign, xsign in itertools.product(
                ymagss, xmagss, [-1, +1], [-1, +1]
        ):
            yvals = ymags * ysign
            xvals = xmags * xsign
            _poss.append([yvals, xvals])
        poss = np.array(_poss)
        valid_ind = (
                (ymin <= poss[:, 0, :])
                & (poss[:, 0, :] <= ymax)
                & (xmin <= poss[:, 1, :])
                & (poss[:, 1, :] <= xmax)
        )
        assert np.any(valid_ind)
        valid_ind = np.any(valid_ind, axis=0)
        for pos in np.moveaxis(poss[:, :, valid_ind], -1, 0)[: int(n)]:
            for yval, xval in pos:
                if (ymin <= yval) and (yval <= ymax) and (xmin <= xval) and (xval <= xmax):
                    subI1 = self.extract_overlap_subregion(image1, yval, xval)
                    subI2 = self.extract_overlap_subregion(image2, -yval, -xval)
                    if subI1.size / (sizeX * sizeY) > 0.05 and min(subI1.shape) > 80:  # 在overlap区域, 最少10%的重叠区域
                        ncc_val = self.ncc(subI1, subI2)
                        if ncc_val > _ncc:
                            _ncc = float(ncc_val)
                            y = int(yval)
                            x = int(xval)
                            overlap_img_1 = subI1
                            overlap_img_2 = subI2
        return _ncc, y, x, overlap_img_1, overlap_img_2

    @staticmethod
    def multi_peak_max(PCM):
        """Find the first to n th largest peaks in PCM.

        Parameters
        ---------
        PCM : np.ndarray
            the peak correlation matrix

        Returns
        -------
        rows : np.ndarray
            the row indices for the peaks
        cols : np.ndarray
            the column indices for the peaks
        vals : np.ndarray
            the values of the peaks
        """
        row, col = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
        vals = PCM[row[::-1], col[::-1]]
        return row[::-1], col[::-1], vals

    @staticmethod
    def pcm(image1, image2):
        """Compute peak correlation matrix for two images.

        Parameters
        ---------
        image1 : np.ndarray
            the first image (the dimension must be 2)

        image2 : np.ndarray
            the second image (the dimension must be 2)

        Returns
        -------
        PCM : np.ndarray
            the peak correlation matrix
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        F1 = np.fft.fft2(image1)
        F2 = np.fft.fft2(image2)
        FC = F1 * np.conjugate(F2)
        return np.fft.ifft2(FC / np.abs(FC)).real.astype(np.float32)

    @staticmethod
    def ncc(image1, image2):
        """Compute the normalized cross correlation for two images.

        Parameters
        ---------
        image1 : np.ndarray
            the first image (the dimension must be 2)

        image2 : np.ndarray
            the second image (the dimension must be 2)

        Returns
        -------
        ncc : Float
            the normalized cross correlation
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        image1 = image1.flatten()
        image2 = image2.flatten()
        n = np.dot(image1 - np.mean(image1), image2 - np.mean(image2))
        d = np.linalg.norm(image1) * np.linalg.norm(image2)
        return n / d

    @staticmethod
    def extract_overlap_subregion(image, y: int, x: int):
        """Extract the overlapping subregion of the image.

        Parameters
        ---------
        image : np.ndarray
            the image (the dimension must be 2)
        y : Int
            the y (second last dim.) position
        x : Int
            the x (last dim.) position
        Returns
        -------
        subimage : np.ndarray
            the extracted subimage
        """
        sizeY = image.shape[0]
        sizeX = image.shape[1]
        assert (np.abs(y) < sizeY) and (np.abs(x) < sizeX)
        # clip x to (0, size_Y)
        h_start = int(max(0, min(y, sizeY, key=int), key=int))
        # clip x+sizeY to (0, size_Y)
        h_end = int(max(0, min(y + sizeY, sizeY, key=int), key=int))
        w_start = int(max(0, min(x, sizeX, key=int), key=int))
        w_end = int(max(0, min(x + sizeX, sizeX, key=int), key=int))
        return image[h_start:h_end, w_start:w_end]

    @staticmethod
    def show_3d(image):
        from matplotlib import cm
        import matplotlib.pyplot as plt
        h, w = image.shape
        X = np.arange(0, w, 1)
        Y = np.arange(0, h, 1)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, image, cmap=cm.coolwarm)
        plt.show()


def main():
    dst_path = r"D:\data\stitching_test_data\Cy5_Image_Tiles\Cy5_Image_Tiles\img_Cy5_r006_c001.tif"
    src_path = r"D:\data\stitching_test_data\Cy5_Image_Tiles\Cy5_Image_Tiles\img_Cy5_r006_c002.tif"
    dst_img = cv.imread(dst_path, 0)
    src_img = cv.imread(src_path, 0)
    match = FFTMatcher()
    match.horizontal = True
    match.overlap = 0.13
    result0 = match.neighbor_match(dst_img, src_img)
    result1 = match.neighbor_match_v1(dst_img, src_img)
    result2 = match.neighbor_match_v2(dst_img, src_img)
    print(result0, result1, result2)


if __name__ == '__main__': main()
