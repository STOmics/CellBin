import os.path as osp
import glog


class DetectorModelConfig(object):
    def __init__(self):
        curr_dir = osp.dirname((osp.abspath(__file__)))
        model_dir = osp.join(osp.dirname(osp.dirname(curr_dir)), "models")
        weights_name = "ST_TP_v1.1.0_5000_5_7.weights"
        self.model_path = osp.join(model_dir, weights_name)
        glog.info(f"Cross detector using {weights_name}")
        self.net_path = osp.join(model_dir, 'ST_TP.cfg')
        self.net_width = 1536
        self.net_height = 1024
        self.GPU = False
        self.confidence_score = 0.85
        self.mns_threshold = 0.1


class ClassificationModelConfig(object):
    def __init__(self):
        curr_dir = osp.dirname((osp.abspath(__file__)))
        model_name = "cv_enhance"
        model_dir = osp.join(osp.dirname(curr_dir), "track_cross_classification", "classification_models", model_name)
        self.check_point_path = osp.join(model_dir, "best_model.pth")
        glog.info(f"Classification using {model_name}")


class LineDetectorModelConfig(object):
    def __init__(self):
        curr_dir = osp.dirname((osp.abspath(__file__)))
        model_name = "sold2_wireframe.tar"
        self.multiscale = False
        self.ckpt_path = osp.join(osp.dirname(curr_dir), "sold2", "ckpt", model_name)
        self.config_path = osp.join(osp.dirname(curr_dir), "sold2", "config", "export_line_features.yaml")
