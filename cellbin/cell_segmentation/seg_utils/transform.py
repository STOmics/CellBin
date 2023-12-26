import multiprocessing as mp
import cv2
from skimage import filters
from skimage.morphology import disk
#import cupy
from skimage.filters import median #cucim.
from skimage.morphology import disk as disk_cu #cucim.
import glog
from seg_utils.utils import split_preproc, merge_preproc
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity


def percentile_threshold(img, percentile=99.9):
    """Threshold an image to reduce bright spots
    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    current_img = np.copy(img)
    non_zero_vals = current_img[np.nonzero(current_img)]

    # only threshold if channel isn't blank
    if len(non_zero_vals) > 0:
        img_max = np.percentile(non_zero_vals, percentile)

        # threshold values down to max
        threshold_mask = current_img > img_max
        current_img[threshold_mask] = img_max

    return current_img


def histogram_normalization(img, kernel_size=None):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    # if not np.issubdtype(image.dtype, np.floating):
    #     logging.info('Converting image dtype to float')
    img = img.astype('float32')

    sample_value = img[(0,) * img.ndim]
    if (img == sample_value).all():
        return np.zeros_like(img)

    # X = rescale_intensity(X, out_range='float')
    img = rescale_intensity(img, out_range=(0.0, 1.0))
    img = equalize_adapthist(img, kernel_size=kernel_size)

    return img


class transform(object):
    """ Transform """

    def __init__(self, image):
        self.image = image

    def median_filter_single(self, image):

        m_image = filters.median(image, disk(50))
        m_image = cv2.subtract(image, m_image)
        return m_image

    def median_filter_in_pool(self, image_list, images, processes=20):
        with mp.Pool(processes=processes) as p:
            for img in image_list:
                median_image = p.apply_async(self.median_filter_single, (img,))
                images.append(median_image)
            p.close()
            p.join()

    def denoise_with_median(self, processes=20):
        glog.info('median filter using cpu')
        image_list, m_x_list, m_y_list = split_preproc(self.image, 1000, 100)
        images = []
        self.median_filter_in_pool(image_list, images, processes)
        filter_image = merge_preproc(images, m_x_list, m_y_list, self.image.shape, 100)
        return filter_image

    def denoise_with_median_cuda(self):
        glog.info('median filter using gpu')
        image_cp = cupy.asarray(self.image)
        median_image = median(image_cp, disk_cu(50))
        median_image = np.asarray(median_image.get())
        filter_image = cv2.subtract(self.image, median_image)
        return filter_image

    def mesmer_prepro(self, percentile=99.9, kernel_size=256):
        image = percentile_threshold(self.image, percentile)
        image = histogram_normalization(image, kernel_size)
        image = (image * 255).astype(np.uint8)
        return image

