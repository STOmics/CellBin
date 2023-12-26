import os
import numpy as np


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


def rc_key(row: int, col: int): return '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))


def get_rc_from_image_map(imap):
    rc = np.array([k.split('_') for k in list(imap.keys())], dtype=int)
    c = int(np.max(rc[:, 1]) - np.min(rc[:, 1]) + 1)
    r = int(np.max(rc[:, 0]) - np.min(rc[:, 0]) + 1)
    return r, c

