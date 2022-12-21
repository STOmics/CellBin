import os
import glog
from labelling.correct import adjust
from segmentation.tissue import tissue_cut
from segmentation.cell import cell_seg
from stitching import stitch

from absl import flags
from absl import app


class CellBin(object):
    def __init__(self, output_path: str=None):
        self.image = None
        self.gene_matrix = None
        self.output_path = output_path
        self.prefix = None

    def _is_ok(self, dirs):
        for it in dirs:
            if not os.path.exists(it):
                glog.warn('The path {} not exists'.format(it))
                if it == self.output_path:
                    assert os.path.isdir(self.output_path)
                    glog.info('Try to create a new output path {}'.format(it))
                    os.makedirs(self.output_path)

    def cell_label(self, registration_image: str, gene_matrix: str):
        self.image = registration_image
        self.gene_matrix = gene_matrix
        self.prefix = os.path.basename(registration_image).split('.')[0]
        self._is_ok([self.image, self.gene_matrix, self.output_path])
        glog.info('Next run: TissueSeg -> CellSeg -> CellLabelling.')
        tissue_path = os.path.join(self.output_path, '{}_tissue_mask.tif'.format(self.prefix))
        tissue_cut(input=self.image, output=tissue_path)
        cell_path = os.path.join(self.output_path, '{}_cell_mask.tif'.format(self.prefix))
        cell_seg(input=self.image, output=cell_path)
        adjust(way='fast', mask_file=cell_path, matrix_file=self.gene_matrix, output=self.output_path)
        glog.info('Cell Labelling finished.')

    def stitching(self, fovs: str):
        self._is_ok([fovs, self.output_path])
        stitch.stitch(fovs, self.output_path)

"""
python .\cell_bin.py \
--image_path D:\code\mine\github\StereoCell\data\SS2000_regist.tif \
--matrix_path D:\code\mine\github\StereoCell\data\SS2000.gem.gz  \
--out_path D:\code\mine\github\StereoCell\data
"""


def main(argv):
    cb = CellBin(FLAGS.out_path)
    cb.cell_label(FLAGS.image_path, FLAGS.matrix_path)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('image_path', '', 'image path')
    flags.DEFINE_string('matrix_path', '', 'gem file')
    flags.DEFINE_string('out_path', '', 'output path')
    app.run(main)
