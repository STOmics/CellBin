import os
import copy
import sys
import glog
import tifffile
try: from .tools import uity
except: from tools import uity
import numpy as np
from absl import flags, app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import controller.processing


class TissueCut(object):
    def __init__(self, gpu="-1", num_threads=0):
        """
        :param img_type:ssdna，rna
        :param model_path:path of weights
        """
        self._WIN_SIZE = None
        self._model_path = None
        self.net_cfg()
        self._gpu = gpu
        self._model = None
        self._num_threads = num_threads
        self._init_model()

    def net_cfg(self, cfg='weights.json'):
        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg)
        import json
        with open(cfg, 'r') as fd:
            dct = json.load(fd)
        self._WIN_SIZE = dct['tissue']['input']
        self._model_path = dct['tissue']['weights_path']
        if not os.path.exists(self._model_path):
            glog.error('Not found weights file in {}.'.format(self._model_path))
        else: glog.info(f"Start load weights from {self._model_path}")

    def _init_model(self):
        from net.onnx_net import cl_onnx_net
        self._model = cl_onnx_net(self._model_path, self._gpu, self._num_threads)

    def f_predict(self, img):
        """
        :param img:CHANGE
        :return: Model input image, mask
        """
        img = np.squeeze(img)
        src_shape = img.shape[:2]

        img = controller.processing.f_tissue_preprocess(img, self._WIN_SIZE)
        pred = self._model.f_predict(copy.deepcopy(img))
        pred = controller.processing.f_post_process(pred)
        pred = uity.f_resize(pred, src_shape)
        return img, pred


def tissue_cut(input: str, output: str, gpu: str=-1, num_threads: int=0):
    if input is None or output is None:
        print("please check your parameters")
        return

    img = tifffile.imread(input)
    sg = TissueCut(gpu=gpu, num_threads=int(num_threads))
    glog.info(f"Weights loaded, start predict.")
    img, pred = sg.f_predict(img)
    glog.info(f"Predict finish, start write.")
    tifffile.imwrite(output, pred)
    glog.info(f"Work Finished.")


def main(argv):
    tissue_cut(input=FLAGS.input,
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
