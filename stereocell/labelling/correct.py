from absl import flags
from absl import app
try:
    from .gmm_correct import GMMCorrect
    from .fast_correct import FastCorrect
except:
    from gmm_correct import GMMCorrect
    from fast_correct import FastCorrect
import glog


def adjust(way, mask_file, matrix_file, output, radius=50, process=10, threshold=20):
    if way == 'fast': correct = FastCorrect()
    elif way == 'GMM':
        correct = GMMCorrect()
        correct.set_radius(radius)
        correct.set_process(process)
        correct.set_threshold(threshold)
    else: glog.error('Unknown Correct Type.')
    glog.info('Correct Type <{}> selected.'.format(way))
    correct.set_output(output)
    correct.creat_cell_gxp(matrx_path=matrix_file, cell_mask_path=mask_file)
    correct.adjust()


def main(argv):
    adjust(FLAGS.way,
           FLAGS.mask_path,
           FLAGS.matrix_path,
           FLAGS.out_path,
           radius=FLAGS.radius,
           process=FLAGS.process,
           threshold=FLAGS.threshold)


"""
python .\correct.py \
--mask_path D:\\code\\mine\\github\\StereoCell\\data\\cell_mask.tif \
--matrix_path D:\\code\\mine\\github\\StereoCell\\data\\gene.txt
--out_path D:\\code\\mine\\github\\StereoCell\\data
"""
if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('mask_path', '', 'cell mask')
    flags.DEFINE_string('matrix_path', '', 'gem file')
    flags.DEFINE_string('out_path', '', 'output path')
    flags.DEFINE_integer('radius', 50, 'Local area.', lower_bound=0)
    flags.DEFINE_integer('process', 3, 'Count of process.', lower_bound=0)
    flags.DEFINE_integer('threshold', 20, 'threshold.', lower_bound=0)
    flags.DEFINE_enum('way', 'GMM', ['fast', 'GMM'], 'Way of adjust.')
    app.run(main)
