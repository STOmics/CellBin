import glob
import os
import sys
import h5py
import glog
import pandas as pd
import getpass
import numpy as np
_cellbin_ = os.path.join(os.path.split(os.path.abspath(__file__))[0], '..')
sys.path.append(_cellbin_)
from cellbin.utils.util import search_files
from cellbin.stitching import Stitcher
from cellbin.registration import Registration
from cellbin.registration.gene_info import GeneInfo
import tifffile
import argparse
import shutil


def image_map(images):
    """
    Args:
        images: List of images
    Returns: dict -> Used to describe the number of row and column indexes that
            each file belongs to in the scanning space
    """
    dct = dict()
    for i in images:
        file_name = os.path.split(i)[-1]
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
        x_str = xy[1]
        y_str = xy[0]

        dct['{}_{}'.format(y_str.zfill(4), x_str.zfill(4))] = i
    return dct


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


class Regist(object):
    """ Registration module """

    def __init__(self):
        """Initialize StereoCell pipeline.

        Args:
          _image_path: Storage path of stain image data.
          _output_path: Result of StereoCell will be saved to.
          _stereo_chip: STOmics chip number.
          _stereo_chip_mode: Cycle interval of STOmics chip.
          _gene_matrix: Gene matrix generate by stereo-seq, Usually a file in gef or gem format.

          _template: Total anchor points of mosaic image.
          _scale_x, _scale_y: The XY-direction scale factor of the stain image relative to the gene matrix.
          _rotation: The rotation factor of the stain image relative to the gene matrix.

        """
        self._image_path = None
        self._output_path = None
        self._output_dir = None
        self._stereo_chip = None
        self._stereo_chip_mode = None
        self._gene_matrix = None

        self._template = None
        self._scale_x = None
        self._scale_y = None
        self._rotation = None

    def set_stereo_chip(self, c): self._stereo_chip = c

    def is_QC_pass(self, ):
        import json
        json_path = glob.glob(os.path.join(self._output_dir, '*.json'))[0]
        with open(json_path, 'r') as fd:
            dct = json.load(fd)
        flag = dct['QCInfo']['QCResultFlag']
        x = (flag == 1 and True or False)
        return x

    def _image_qc(self, ):
        glog.info('Anchor point positioning for subsequent registration process')
        cmd = '{} {} -i {} -o {} -c {} -n {} -e {}'.format(
            sys.executable, '../cellbin/iqc/qc_util.py',
            self._image_path, self._output_dir,
            self._stereo_chip, 'test', getpass.getuser())
        print(cmd)
        os.system(cmd)

    def _h5_path(self, ):
        items = glob.glob(os.path.join(self._output_dir, '*.ipr'))
        return os.path.join(self._output_dir, items[0])

    def _xlsx_path(self, ):
        items = glob.glob(os.path.join(self._output_dir, '*.xlsx'))
        xlsx = os.path.join(self._output_dir, items[0])
        return location2npy(xlsx)

    def _matrix_1bin(self, ):
        items = glob.glob(os.path.join(self._output_dir, '*_matrix.tif'))
        if len(items): return os.path.join(self._output_dir, items[0])
        else: return None

    def _stitching(self, ):
        s = Stitcher()

        ipr = self._h5_path()
        h5 = h5py.File(ipr, 'r')
        scale_x = h5['Register'].attrs['ScaleX']
        scale_y = h5['Register'].attrs['ScaleY']
        rotate = h5['Register'].attrs['Rotation']
        chipno = h5['QCInfo']['TrackDistanceTemplate'][:]
        self._stereo_chip_mode = chipno
        index = h5['Stitch'].attrs['TemplateSource']
        x, y = index.split('_')
        index = '{}_{}'.format(x.zfill(4), y.zfill(4))
        pts = h5['QCInfo']['CrossPoints']
        s.set_global_location(self._xlsx_path())
        pts_ = dict()
        for k in pts.keys():
            r, c = k.split('_')
            key = '{}_{}'.format(r.zfill(4), c.zfill(4))
            pts_[key] = np.array(pts[k][:], dtype=float)
        self._template, _ = \
            s.template(pts=pts_, scale_x= 1 / scale_x, scale_y= 1 / scale_y, rotate=rotate,
                   chipno=self._stereo_chip_mode, index=index, output_path=self._output_dir)
        self._template = np.array(self._template)
        self._scale_x, self._scale_y, self._rotation = _

    def _registration(self, ):
        glog.info('Registration, (Fixed: Gene matrix), (Moving: Image)')

        if self._matrix_1bin() is not None:
            glog.info('Load matrix bin-1 data from {}'.format(self._matrix_1bin()))
            gene_mat = tifffile.imread(self._matrix_1bin())
        else:
            gi = GeneInfo(gene_file=self._gene_matrix, chip_mode=None)
            gene_mat = gi.gene_mat
            matrix_path = os.path.join(self._output_dir, '{}_matrix.tif'.format(self._stereo_chip))
            tifffile.imwrite(matrix_path, gene_mat, compression="zlib", compressionargs={"level": 8})
            glog.info('Save matrix bin-1 data to {}'.format(matrix_path))

        fov_stitched = tifffile.imread(self._image_path)
        r = Registration()
        r.mass_registration_stitch(fov_stitched=fov_stitched, vision_image=gene_mat,
                                   chip_template=self._stereo_chip_mode, track_template=self._template,
                                   scale_x=1 / self._scale_x, scale_y=1 / self._scale_y, rotation=self._rotation, flip=True)
        r.transform_to_regist()
        tifffile.imwrite(self._output_path, r.regist_img)
        shutil.rmtree(self._output_dir)

    def run(self, image: str, output: str, stereo_chip: str, gem=None):
        """ Module of Registration """
        self._image_path = image
        self._output_path = output
        self._output_dir = os.path.join(os.path.dirname(output), 'tmp')
        if not os.path.exists(self._output_dir): os.makedirs(self._output_dir)
        self._stereo_chip = stereo_chip
        self._gene_matrix = gem
        glog.info('Start RUN StereoCell registration')
        self._image_qc()
        if self.is_QC_pass():
            glog.warn('Image QC pass')
            self._stitching()
            if gem is None: glog.warn('Miss gene matrix, will finished the pipeline')
            self._gene_matrix = gem
            self._registration()
        else:
            glog.warn('The track point detection failed and the follow-up process could not be completed')


""" Usage
python .\registration.py --input D:\data\test\SS200000135TL_D1 --output D:\data\test\paper --matrix D:\data\test\SS200000135TL_D1.gem.gz --chipno SS200000135TL_D1
"""


def main(args, para):
    # log_file = time.strftime("%Y-%m-%d-%H%M%S.log", time.localtime())
    # handler_log = logging.FileHandler(filename=os.path.join(output, 'stereocell-{}'.format(log_file)), mode='w')
    # handler_log.setFormatter(glog.GlogFormatter())
    # logging.basicConfig(handlers=[handler_log], level=logging.INFO)

    # input = r'D:\data\test\SS200000135TL_D1'
    # output = r'D:\data\test\paper'
    # chip_no = 'SS200000135TL_D1'
    # gem_file = r'D:\data\test\SS200000135TL_D1.gem.gz'

    input = args.image_file
    output = args.output_file
    chip_no = args.chip_no
    gem_file = args.gene_exp_data

    p = Regist()
    p.run(image=input, output=output, stereo_chip=chip_no, gem=gem_file)


"""
python registration.py -i D:\data\test\paper\stitched.tif -g D:\data\test\SS200000135TL_D1.gem.gz -c SS200000135TL_D1 -o D:\data\test\paper2\registered_image.tif
"""
if __name__ == '__main__':
    usage="""Registration (StereoCell)"""
    PROG_VERSION='v0.0.1'

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-i", "--image_file", action="store", dest="image_file", type=str, required=True, help="Input image dir.")
    parser.add_argument("-g", "--gene_exp_data", action="store", dest="gene_exp_data", type=str, required=True, help="Input gene matrix.")
    parser.add_argument("-c", "--chip_no", action="store", dest="chip_no", type=str, required=True, help="Stereo-seq chip No.")
    parser.add_argument("-o", "--output_file", action="store", dest="output_file", type=str, required=True, help="Result output dir.")
    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
