from os import walk
from os.path import splitext, join, basename
import cv2 as cv
import tifffile
import numpy as np
import pandas as pd
from json import loads, dumps, dump, load


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = splitext(f)
            if ext in exts:
                files_.append(join(root, f))
    return files_


def filename2index(file_name, style, row_len=None):
    file_name = basename(file_name)
    if style.lower() == 'motic':
        # x_str, y_str = splitext(file_name)[0].split('_')
        tags = splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
        x_str = xy[0]
        y_str = xy[1]
        return [int(y_str), int(x_str)]
    elif style.lower() == 'zeiss':
        name, info, tail = splitext(file_name)[0].split('_')
        x_start, x_y, y_len = info.split('-')
        x_start = int(x_start.split('x')[1])
        x_len, y_start = x_y.split('y')
        x_len = int(x_len)
        y_start = int(y_start)
        y_len = int(y_len.split('m')[0])
        overlap_x = int(x_len * 0.9)
        overlap_y = int(y_len * 0.9)
        i = round(x_start / overlap_x)
        j = round(y_start / overlap_y)
        return [i, j]
    elif style.lower() == 'leica':
        prefix, num = splitext(file_name)[0].split('--Stage')
        x = int(int(num) / row_len)
        y = int(int(num) % row_len)
        return [x, y]
    else:
        return None


def img_read(file_path):
    if file_path == 'None':
        return None
    else:
        arr = cv.imread(file_path, -1)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr


def img_read3(file_path):
    if file_path == 'None':
        return None
    else:
        return cv.imread(file_path, -1)


def mat_channel(mat):
    if mat.ndim == 2:
        return 1
    else:
        return 3


def mat_deepth(mat):
    c = mat_channel(mat)
    if c == 1: item = mat[0, 0]
    else: item = mat[0, 0, 0]
    tp = type(item)
    if tp == 'np.uint8': return int(c * 8)
    elif tp == 'np.uint16': return int(c * 16)
    else: return 32


def tiff_write(file_name, mat):
    # opencv BGR -> RGB.
    if mat.ndim == 3: mat = mat[:, :, (2, 1, 0)]
    tifffile.imwrite(file_name, mat)
    # cv.imwrite(file_name, mat,
    #            ((int(cv.IMWRITE_TIFF_RESUNIT), 2,
    #             int(cv.IMWRITE_TIFF_COMPRESSION), 1,
    #             int(cv.IMWRITE_TIFF_XDPI), 300,
    #             int(cv.IMWRITE_TIFF_YDPI), 300)))


def locate(x, start, end):
    if x < start:
        return 0
    elif x >= end:
        return int(end - 1)
    else:
        return int(x)


def npy2xlsx(dct, xlsx: str, is_int=True):
    writer = pd.ExcelWriter(xlsx)
    for k, v in dct.items():
        if is_int:
            v = np.array(v, dtype=np.int64)
        assert v.ndim == 3
        data = pd.DataFrame(v[:, :, 0])
        data.to_excel(writer, k, startrow=0)
        df_rows = v.shape[0]
        data = pd.DataFrame(v[:, :, 1])
        data.to_excel(writer, k, startrow=df_rows + 1)  # float_format='%.2f'
    writer.save()
    writer.close()


def xlsx2npy(xlsx: str):
    def get_npy(arr):
        h, w = arr.shape
        h_ = int(h / 2)
        mat = np.zeros((h_ - 1, w - 1, 2), dtype=np.int64)
        mat[:, :, 0] = arr[1:h_, 1:]
        mat[:, :, 1] = arr[h_ + 1:, 1:]
        return mat

    pr = pd.read_excel(xlsx, sheet_name=['horizontal_offset', 'vertical_offset'], header=None)
    if np.isnan(pr['vertical_offset'][0][0]):
        col = int(np.array(pr['vertical_offset']).shape[1] / 2) - 1
    else:
        col = int(pr['vertical_offset'][0][0])
    row_offset = get_npy(np.array(pr['horizontal_offset']))
    col_offset = get_npy(np.array(pr['vertical_offset']))
    return {'horizontal_offset': row_offset, 'vertical_offset': col_offset, 'col': col}


def location2npy(xlsx: str):
    def get_npy(arr):
        h, w = arr.shape
        h_ = int(h / 2)
        mat = np.zeros((h_ - 1, w - 1, 2), dtype=np.int64)
        mat[:, :, 0] = arr[1:h_, 1:]
        mat[:, :, 1] = arr[h_ + 1:, 1:]
        return mat

    table_name = 'location'
    pr = pd.read_excel(xlsx, sheet_name=[table_name], header=None)
    location = get_npy(np.array(pr[table_name]))
    return location


# def npy2xlsx(dct, xlsx: str, is_int=True):
#     try:
#         with open(xlsx, 'w', newline='') as fd:
#             csv_w = csv.writer(fd)
#             for k, v in dct.items():
#                 if is_int:
#                     v = np.array(v, dtype=np.int64)
#                 assert v.ndim == 3
#                 data = v[:, :, 0]
#                 h, w = data.shape
#                 header = [''] + ['C{}'.format(i) for i in range(w)]
#                 csv_w.writerow(header)
#                 for i in range(h):
#                     csv_w.writerow([i] + data[i, :].tolist())
#
#                 csv_w.writerow('')
#
#                 data = v[:, :, 1]
#                 csv_w.writerow(header)
#                 for i in range(h):
#                     csv_w.writerow([i] + data[i, :].tolist())
#
#                 # csv_w.writerows(data)
#             fd.close()
#     except Exception as e:
#         glog.info("Write an CSV file to path: {}, Case: {}".format(xlsx, e))


class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, item):
        return None


def json_serialize(obj, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as fd:
        str_dct = dumps(obj, default=lambda o: o.__dict__)
        dump(loads(str_dct), fd, indent=2, ensure_ascii=False)


def json_2_dict(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as fd:
        return load(fd)


# about dict
def dict_deserialize(dct: dict) -> JSONObject:
    str_dct = dumps(dct)
    return loads(str_dct, object_hook=JSONObject)


# about json
def json_deserialize(file_path: str) -> JSONObject:
    dct = json_2_dict(file_path)
    return dict_deserialize(dct)

