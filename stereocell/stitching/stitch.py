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

"""Basic operations of multi-image stitching."""
import glog
import numpy as np
from absl import flags, app
from stitch import neighbor2_stitcher
import os

PROG_VERSION = '0.1.0'
PROG_DATE = '2022-07-26'


class DataLoader(object):
    def __init__(self):
        self.fov_path = None
        self._support = ['.tif', '.tiff', '.png', '.jpg']
        self.data_pool = None

    @staticmethod
    def search_files(file_path, exts):
        files_ = list()
        for root, dirs, files in os.walk(file_path):
            if len(files) == 0: continue
            for f in files:
                fn, ext = os.path.splitext(f)
                if ext in exts: files_.append(os.path.join(root, f))
        return files_

    @staticmethod
    def _parse_index(file_name):
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
        x_str = xy[0]
        y_str = xy[1]
        return [int(y_str), int(x_str)]

    def _r0c0(self, fovs):
        names = list()
        for fov in fovs:
            c, r = self._parse_index(os.path.basename(fov))
            names.append([r, c])
        grid = np.array(names, dtype=int)
        r0, c0 = [np.min(grid[:, 0]), np.min(grid[:, 1])]
        return r0, c0

    def load(self, src):
        self.fov_path = src
        # just support format: row_col.tif, other format can be modified by imageQC.
        fovs = self.search_files(self.fov_path, self._support)
        r0, c0 = self._r0c0(fovs)
        if not len(fovs): return 1

        self.data_pool = dict()
        for fov in fovs:
            c, r = self._parse_index(os.path.basename(fov))
            c, r = [c - c0, r - r0]
            self.data_pool['{}_{}'.format(str(r).zfill(4), str(c).zfill(4))] = fov

        return self.data_pool


def stitch(input: str, output: str, overlap=0.12):
    stitcher = neighbor2_stitcher.Neighbor2(overlap=overlap)
    dl = DataLoader()
    dct = dl.load(input)
    import time
    t0 = time.time()
    stitcher.stitching(dct)
    t1 = time.time()
    glog.info('Stitching time is {} ms'.format(round(1000 * (t1 - t0))))
    stitcher.save_mosaic(output)
    t2 = time.time()
    glog.info('Mosaic generate time is {} ms'.format(round(1000 * (t2 - t1))))
    stitcher.export_loc(os.path.dirname(output))


def main(argv): stitch(input=FLAGS.input, output=FLAGS.output, overlap=FLAGS.overlap)


"""
python .\stitch.py --input D:\data\studio_test\studio_test\SS200000135TL_D1_Au\SS200000135TL_D1 --output D:\data\studio_test\studio_test\SS200000135TL_D1_Au
--overlap 0.12 
"""
if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('input', '', 'FOV images storage location')
    flags.DEFINE_string('output', '', 'Save path of stitch result files')
    flags.DEFINE_float('overlap', 0.12, 'FOV images Overlap.', lower_bound=0)
    app.run(main)
