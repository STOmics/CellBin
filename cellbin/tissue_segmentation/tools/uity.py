import cv2
import numpy as np
from PIL import Image
import copy


def f_padding(img, shape, mode):
    h, w = img.shape[: 2]
    win_h, win_w = shape[:2]
    img = np.pad(img, ((0, abs(win_h - h)), (0, abs(win_w - w))), mode)
    return img


def f_resize(img, shape=(1024, 2048), mode="NEAREST"):
    imode = Image.NEAREST
    if mode == "BILINEAR":
        imode = Image.BILINEAR
    elif mode == "BICUBIC":
        imode = Image.BICUBIC
    elif mode == "LANCZOS":
        imode = Image.LANCZOS
    elif mode == "HAMMING":
        imode = Image.HAMMING
    elif mode == "BOX":
        imode = Image.BOX
    if img.dtype != 'uint8':
        imode = Image.NEAREST
    img = Image.fromarray(img)
    img = img.resize((shape[1], shape[0]), imode)
    img = np.array(img).astype(np.uint8)
    return img


def f_ij_16_to_8(img, chunk_size=1000):
    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape[:2], np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = copy.deepcopy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst


def f_ij_auto_contrast(img):
    limit = img.size / 10
    threshold = img.size / 5000
    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256
    hist, _ = np.histogram(img.flatten(), 256, [0, bit_max])
    hmin = 0
    hmax = bit_max - 1
    for i in range(1, len(hist) - 1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(len(hist) - 2, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    if hmax > hmin:
        hmax = int(hmax * bit_max / 256)
        hmin = int(hmin * bit_max / 256)
        img[img < hmin] = hmin
        img[img > hmax] = hmax
        cv2.normalize(img, img, 0, bit_max - 1, cv2.NORM_MINMAX)
    return img


def f_fill_hole(im_in):
    ''' Hole filling for binary images '''
    im_floodfill = cv2.copyMakeBorder(im_in, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0])
    # im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill[2:-2, 2:-2])
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out


if __name__ == '__main__':
    pass
