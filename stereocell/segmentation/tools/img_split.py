import numpy as np
from tools.uity import f_padding
import math


def _f_split(img, win_shape, box_lst, overlap=100):
    h, w = img.shape[:2]
    win_h, win_w = win_shape[:2]
    y_nums = math.ceil(h / (win_h - overlap))
    x_nums = math.ceil(w / (win_w - overlap))
    for y_temp in range(y_nums):
        for x_temp in range(x_nums):
            x_begin = int(max(0, x_temp * (win_w - overlap)))
            y_begin = int(max(0, y_temp * (win_h - overlap)))
            x_end = int(min(x_begin + win_w, w))
            y_end = int(min(y_begin + win_h, h))
            if y_begin >= y_end or x_begin >= x_end:
                continue
            box_lst.append([y_begin, y_end, x_begin, x_end])
    return


def _f_get_batch_input(img, batch_box, win_shape, padding, padding_mode):
    batch_input = []
    for box in batch_box:
        y_begin, y_end, x_begin, x_end = box
        img_win = img[y_begin: y_end, x_begin: x_end]
        if padding:
            img_win = f_padding(img_win, win_shape, padding_mode)
        batch_input.append(img_win)
    return batch_input


def _f_set_img(ret, box, img_win, overlap):
    h, w = ret.shape[:2]
    win_y_begin, win_x_begin = 0, 0
    y_begin, y_end, x_begin, x_end = box
    if overlap != 0:
        if y_begin != 0:
            y_begin = y_begin + overlap // 2
            win_y_begin = win_y_begin + overlap // 2
        if x_begin != 0:
            x_begin = x_begin + overlap // 2
            win_x_begin = win_x_begin + overlap // 2
        if y_end != h:
            y_end = y_end - overlap // 2
        if x_end != w:
            x_end = x_end - overlap // 2
    ret[y_begin: y_end, x_begin: x_end] = img_win[win_y_begin: win_y_begin + y_end - y_begin,
                                          win_x_begin: win_x_begin + x_end - x_begin]
    return ret


def _f_combine(img, win_shape, overlap, editable, padding, padding_mode, dtype, batch_size, box_lst, fun, *args):
    ret = img

    if len(box_lst) < 2:
        if padding:
            ret = f_padding(img, win_shape, padding_mode)
        ret = fun(ret, *args)
        return ret

    if not editable:
        ret = np.zeros(img.shape, dtype=dtype)

    for i in range(0, len(box_lst), batch_size):
        batch_box = box_lst[i:min(i + batch_size, len(box_lst))]
        batch_input = _f_get_batch_input(img, batch_box, win_shape, padding, padding_mode)
        batch_output = []
        if batch_size > 1:
            batch_output = fun(batch_input, *args)
        else:
            batch_output = [fun(batch_input[0], *args)]

        for box, pred in zip(batch_box, batch_output):
            ret = _f_set_img(ret, box, pred, overlap)
    return ret


def f_run_with_split(img, win_shape, overlap, padding, padding_mode, editable, batch_size, dtype, fun, *args):
    box_lst = []
    _f_split(img, win_shape, box_lst, overlap)
    img = _f_combine(img, win_shape, overlap, editable, padding, padding_mode, dtype, batch_size, box_lst, fun, *args)
    return img
