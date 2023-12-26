import numpy as np
import cv2
from skimage import filters


def _f_th(img, theta):
    img[img >= theta] = 255
    img[img < theta] = 0
    return


def f_th_otsu(img):
    try:
        theta = filters.threshold_otsu(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_mean(img):
    try:
        theta = filters.threshold_mean(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_li(img):
    try:
        theta = filters.threshold_li(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_triangle(img):
    try:
        theta = filters.threshold_triangle(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_isodata(img):
    try:
        theta = filters.threshold_isodata(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_minimum(img):
    try:
        theta = filters.threshold_minimum(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_yen(img):
    try:
        theta = filters.threshold_yen(img)
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_shanbhag(img):
    try:
        H = cv2.calcHist([img], [0], None, [256], [0, 256])

        total = 0
        for ih in range(0, len(H)):
            total += H[ih]

        norm_histo = [0.0] * len(H)
        for ih in range(0, len(H)):
            norm_histo[ih] = H[ih] / total

        P1 = [0.0] * len(H)
        P2 = [0.0] * len(H)
        P1[0] = norm_histo[0]
        P2[0] = 1.0 - P1[0]
        for ih in range(1, len(H)):
            P1[ih] = P1[ih - 1] + norm_histo[ih]
            P2[ih] = 1.0 - P1[ih]

        first_bin = 0
        for ih in range(0, len(H)):
            if not (abs(P1[ih]) < 2.220446049250313E-16):
                first_bin = ih
                break

        last_bin = len(H) - 1
        for ih in range(len(H) - 1, first_bin - 1, -1):
            if not (abs(P2[ih]) < 2.220446049250313E-16):
                last_bin = ih
                break

        theta = -1
        min_ent = 1.7976931348623157e+308

        for it in range(first_bin, last_bin):
            ent_back = 0.0
            term = 0.5 / P1[it]
            for ih in range(1, it):
                ent_back -= norm_histo[ih] * np.log(1.0 - term * P1[ih - 1])
            ent_back *= term
            ent_obj = 0.0
            term = 0.5 / P2[it]
            for ih in range(it + 1, len(H)):
                ent_obj -= norm_histo[ih] * np.log(1.0 - term * P2[ih])
            ent_obj *= term
            tot_ent = abs(ent_back - ent_obj)
            if tot_ent < min_ent:
                min_ent = tot_ent
                theta = it
        _f_th(img, theta)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


def f_th_max_entropy(img):
    def entp(x):
        temp = np.multiply(x, np.log(x))
        temp[np.isnan(temp)] = 0
        return temp

    try:
        H = cv2.calcHist([img], [0], None, [256], [0, 256])
        H = H / np.sum(H)

        theta = np.zeros(256)
        Hf = np.zeros(256)
        Hb = np.zeros(256)

        for T in range(1, 255):
            a = H[:T - 1]
            b = np.sum(H[1:T - 1])
            if b == 0:
                Hf[T] = 0
            else:
                Hf[T] = - np.sum(entp(np.divide(a, b)))

            a = H[T:]
            b = np.sum(H[T:])
            if b == 0:
                Hb[T] = 0
            else:
                Hb[T] = - np.sum(entp(np.divide(a, b)))
            theta[T] = Hf[T] + Hb[T]

        theta_max = np.argmax(theta)
        _f_th(img, theta_max)
    except:
        return np.zeros(img.shape[:2], np.uint8)
    return img


if __name__ == '__main__':
    import uity
    import tifffile

    img = uity.read_img("test.tif")
    tifffile.imwrite("./th/th_yen.tif", th_yen(img))
    tifffile.imwrite("./th/th_mean.tif", th_mean(img))
    tifffile.imwrite("./th/th_otsu.tif", th_otsu(img))
    tifffile.imwrite("./th/th_isodata.tif", th_isodata(img))
    tifffile.imwrite("./th/th_triangle.tif", th_triangle(img))
    tifffile.imwrite("./th/th_shanbhag.tif", th_shanbhag(img))
    tifffile.imwrite("./th/th_max_entropy.tif", th_max_entropy(img))
    tifffile.imwrite("./th/th_li.tif", th_li(img))

    pass