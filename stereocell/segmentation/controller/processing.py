import numpy as np
import tools.uity
import tools.preprocessing
from tools.deep_watershed import f_deep_watershed


def f_preprocess(img, ):
    img = np.squeeze(img)
    img = tools.uity.f_ij_16_to_8(img)
    img = tools.preprocessing.f_percentile_threshold(img)
    img = tools.preprocessing.f_histogram_normalization(img, 128).astype(np.float32)
    return img


def f_post_process(pred):
    pred = [pred]
    pred = f_deep_watershed(pred, watershed_line=1, fill_holes_threshold=1)
    pred = pred[0, :, :, 0]
    pred[pred > 0] = 1
    pred = np.uint8(pred)
    return pred


from tools.threshold import f_th_li
from skimage.exposure import rescale_intensity


def f_tissue_preprocess(img, tar_size=(256, 256)):
    img = tools.uity.f_ij_16_to_8(img)
    img = tools.uity.f_ij_auto_contrast(img)
    img = tools.uity.f_resize(img, tar_size, "BILINEAR")

    img = img.astype('float32')
    sample_value = img[(0,) * img.ndim]
    if (img == sample_value).all():
        return np.zeros_like(img)
    img = rescale_intensity(img, out_range=(0.0, 1.0))
    return img


def f_tissue_postprocess(pred):
    pred = np.uint8(pred[0] * 255)
    pred = np.squeeze(pred)
    pred = f_th_li(pred)
    return pred

