import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity


def f_percentile_threshold(img, percentile=99.9):
    """Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    # non_zero_vals = img[np.nonzero(img)]
    non_zero_vals = img[img > 0]

    # only threshold if channel isn't blank
    if len(non_zero_vals) > 0:
        img_max = np.percentile(non_zero_vals, percentile)

        # threshold values down to max
        threshold_mask = img > img_max
        img[threshold_mask] = img_max

    return img


def f_histogram_normalization(img, kernel_size=None):
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
