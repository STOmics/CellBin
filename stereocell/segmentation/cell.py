import os
import glog
import tifffile
import numpy as np
from absl import flags, app
try:
    from controller.predict import cl_predict
    from tools.img_split import f_run_with_split
    from controller.processing import f_preprocess
except:
    from .controller.predict import cl_predict
    from .tools.img_split import f_run_with_split
    from .controller.processing import f_preprocess


class CellSeg(object):

    def __init__(self, gpu="-1", num_threads=0):
        """
        :param model_path: path of the CNN model
        """
        self.WIN_SIZE = None
        self._model_path = None
        self._net_cfg()
        self.overlap = 64

        self._gpu = gpu
        self._model = None
        self._sess = None
        self._num_threads = num_threads
        self._f_init_model()

    def _net_cfg(self, cfg='weights.json'):
        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg)
        import json
        with open(cfg, 'r') as fd:
            dct = json.load(fd)
        self.WIN_SIZE = dct['cell']['input']
        self._model_path = dct['cell']['weights_path']
        if not os.path.exists(self._model_path):
            glog.error('Not found weights file in {}.'.format(self._model_path))
        else: glog.info(f"Start load weights from {self._model_path}")

    def _f_init_model(self):
        """

        """
        from net.onnx_net import cl_onnx_net
        self._model = cl_onnx_net(self._model_path, self._gpu, self._num_threads)
        self._sess = cl_predict(self._model)

    def f_predict(self, img):
        """

        :param img:CHANGE
        :return: mask
        """
        img = f_preprocess(img)
        pred = f_run_with_split(img, self.WIN_SIZE, self.overlap, True, 'minimum', False, 1000, np.uint8,
                                self._sess.f_predict)
        return pred


def cell_seg(input: str, output: str, gpu: str=-1, num_threads: int=0):
    if input is None or output is None:
        print("please check your parameters")
        return

    img = tifffile.imread(input)
    sg = CellSeg(gpu=gpu, num_threads=int(num_threads))
    pred = sg.f_predict(img)
    glog.info(f"Predict finish,start write")
    tifffile.imwrite(output, pred, compression="zlib", compressionargs={"level": 8})
    glog.info(f"Work finished.")


"""
python .\tissue.py \
-i D:\data\studio_test\studio_test\SS200000135TL_D1_Au\SS200000135TL_D1\controlD1_0004_0003_2021-11-25_12-26-41-840.tif \
-o D:\data\studio_test\studio_test\SS200000135TL_D1_Au\mask_test_tissue.tif
"""


# def del_all_flags(FLAGS):
#     flags_dict = FLAGS._flags()
#     keys_list = [keys for keys in flags_dict]
#     for keys in keys_list: FLAGS.delattr(keys)


def main(argv):
    cell_seg(input=FLAGS.input,
             output=FLAGS.output,
             gpu=FLAGS.gpu,
             num_threads=FLAGS.num_threads)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('input', '', 'the input img path')
    flags.DEFINE_string('output', '', 'the output file')
    flags.DEFINE_string('gpu', '-1', 'output path')
    flags.DEFINE_integer('num_threads', 0, 'num threads.', lower_bound=0)
    app.run(main)
