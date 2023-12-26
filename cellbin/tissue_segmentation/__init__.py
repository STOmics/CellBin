import os
import copy
import sys
import glog
import tifffile
try: from .tools import uity
except: from tools import uity
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import processing


usage = '''
     limin  %s
     Usage: %s imagePath outPath imageType(1:ssdna; 0:RNA)  method(1:deep; 0:other)
''' % ('2021-07-15', os.path.basename(sys.argv[0]))


class TissueCut(object):
    def __init__(self, weights_file, gpu="-1", num_threads=0):
        """
        :param img_type:ssdna，rna
        :param model_path:path of weights
        """
        self._WIN_SIZE = None
        self._model_path = None
        self.net_cfg(weights_file)
        self._gpu = gpu
        self._model = None
        self._num_threads = num_threads
        self._init_model()

    def net_cfg(self, weights_file):
        self._WIN_SIZE = (512, 512)
        self._model_path = weights_file
        if not os.path.exists(self._model_path):
            glog.error('Not found weights file in {}.'.format(self._model_path))
        else: pass

    def _init_model(self):
        from onnx_net import cl_onnx_net
        self._model = cl_onnx_net(self._model_path, self._gpu, self._num_threads)

    def f_predict(self, img):
        """
        :param img:CHANGE
        :return: Model input image, mask
        """
        img = np.squeeze(img)
        src_shape = img.shape[:2]

        img = processing.f_tissue_preprocess(img, self._WIN_SIZE)
        pred = self._model.f_predict(copy.deepcopy(img))
        pred = processing.f_tissue_postprocess(pred)
        pred = uity.f_resize(pred, src_shape)
        return img, pred


def tissue_cut(input: str, output: str, gpu: str=-1, num_threads: int=0):
    if input is None or output is None:
        print("please check your parameters")
        return

    img = tifffile.imread(input)
    weights = os.path.join(os.path.split(os.path.abspath(__file__))[0], r"../weights/stereocell_bcdu_tissue_512x512_220822.onnx")
    glog.info('Load weights from {}'.format(weights))
    sg = TissueCut(weights_file=weights, gpu=gpu, num_threads=int(num_threads))
    img, pred = sg.f_predict(img)
    glog.info(f"Predict finish, start write.")
    tifffile.imwrite(output, pred, compression="zlib", compressionargs={"level": 8})
    glog.info(f"Work Finished.")


if __name__ == '__main__':
    import sys

    tissue_cut(r"D:\stock\dataset\test\1\registered_image.tif", r"D:\stock\dataset\test\1\out\mask_out.tif")
    sys.exit()

