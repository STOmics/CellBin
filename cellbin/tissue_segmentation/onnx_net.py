import glog
import numpy as np
import onnxruntime
from os import path


class cl_onnx_net(object):
    """ ONNX Net """
    def __init__(self, model_path, gpu="-1", num_threads=0):
        """Initialize ONNX Net.

        Args:
          _providers: CPUExecutionProvider
          _providers_id: -1
          _model_path: Path of weights file.
          _gpu: The serial number of the GPU, -1 means CPU is used.
          _input_name: _input name Layer-0
          _num_threads: The number of threads to use.
        """
        self._providers = ['CPUExecutionProvider']
        self._providers_id = [{'device_id': -1}]
        self._model = None
        self._gpu = gpu
        self._model_path = model_path
        self._input_name = 'input_1'
        self._num_threads = num_threads
        self._f_init()

    def _f_init(self):
        """ Initialize the CNN network """
        if int(self._gpu) > -1:
            self._providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._providers_id = [{'device_id': int(self._gpu)}, {'device_id': -1}]
        else: glog.info('Use CPU to inference the image.')
        self._f_load_model()

    def _f_load_model(self):
        """ Load CNN weights file for inference """
        if path.exists(self._model_path):
            sessionOptions = onnxruntime.SessionOptions()
            if self._gpu == "-1":
                sessionOptions.intra_op_num_threads = self._num_threads

            self._model = onnxruntime.InferenceSession(self._model_path, providers=self._providers,
                                                       provider_options=self._providers_id, sess_options=sessionOptions)
            self._input_name = self._model.get_inputs()[0].name
            glog.info(f"Weight loaded, start predict")
        else:
            glog.warn("Weight file does not exist in {}.".format(self._model_path))

    def f_predict(self, img):
        """ Predict for tissue segmentation """
        pred = self._model.run(None, {self._input_name: np.expand_dims(np.array([img], dtype=np.float32), axis=3)})[0]
        return pred
