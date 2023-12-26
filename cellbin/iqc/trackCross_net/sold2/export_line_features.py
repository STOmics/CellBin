"""
    Export line detections and descriptors given a list of input images.
"""
# from os.path import dirname, join
# import sys
# cur_module = dirname(__file__)
# trackCross = dirname(cur_module)
# sys.path.append(trackCross)
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import glob
import time
import glog

from sold2.experiment import load_config
from sold2.model.line_matcher import LineMatcher
from fov_qc.cross_detector import PreProcess
from sold2.utils.line_util import Line


class LineDetector(object):
    def __init__(self, config_path, multiscale, ckpt_path):
        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.multiscale = multiscale
        self.ckpt_path = ckpt_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        glog.info(f"Line detector model using device -> {self.device}")
        self.line_matcher = None
        self.extension = "_" + "sold2" + ".tif"

    def load_model(self):
        self.line_matcher = LineMatcher(model_cfg=self.config["model_cfg"],
                                        ckpt_path=self.ckpt_path,
                                        device=self.device,
                                        line_detector_cfg=self.config["line_detector_cfg"],
                                        line_matcher_cfg=self.config["line_matcher_cfg"],
                                        multiscale=self.multiscale)

    def post_process(self, line_seg, img):
        adjust_angles = []
        l = Line()
        if img is not None:
            dst = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            dst = None
        for i in range(len(line_seg)):
            p0 = (int(line_seg[i, 0, 1]), int(line_seg[i, 0, 0]))
            p1 = (int(line_seg[i, 1, 1]), int(line_seg[i, 1, 0]))
            if img is not None:
                cv2.line(dst, p0, p1, (0, 0, 255), 1)
            l.init_by_point_pair(p0, p1)
            angle = l.calculate_angle()
            if angle < 0:
                angle_adjust = round(angle + 90)
            else:
                angle_adjust = round(angle)
            adjust_angles.append(angle_adjust)
        return dst, adjust_angles

    def __call__(self, images_list, output_path=None):
        angle_all = []
        p = PreProcess()
        for img, img_name in tqdm(images_list):
            t1 = time.time()
            # img = cv2.imread(img_path, -1)
            img_enhance = p.process(img, multichannel=False)
            img = torch.tensor(img_enhance[None, None] / 255., dtype=torch.float, device=self.device)
            # Run the line detection and description
            ref_detection = self.line_matcher.line_detection(img)
            ref_line_seg = ref_detection["line_segments"]
            # Write the output on disk
            # img_name = os.path.splitext(os.path.basename(img_path))[0]
            if output_path is None:
                img_enhance = None
            combine_image, adjust_angles = self.post_process(ref_line_seg, img_enhance)
            angle_all.extend(adjust_angles)
            if output_path is not None:
                output_file = os.path.join(output_path, img_name + self.extension)
                cv2.imwrite(output_file, combine_image)
            t2 = time.time()
        if len(angle_all) != 0:
            bins = np.bincount(angle_all)
            winner = np.argwhere(bins == np.amax(bins))
            if len(winner) > 1:
                if np.max(winner) - np.min(winner) <= 3:
                    most_frequent_angle = winner.flatten().tolist()[0]
                else:
                    most_frequent_angle = None
            else:
                most_frequent_angle = winner.flatten().tolist()[0]
        else:
            most_frequent_angle = None
        return most_frequent_angle


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Get the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device -> {device}")
    extension = 'sold2'
    extension = "_" + extension + ".tif"
    ckpt_path = r"D:\PycharmProjects\sold2\pretrained_model\sold2_wireframe.tar"
    config_path = r"D:\PycharmProjects\sold2\sold2\config\export_line_features.yaml"
    # config = load_config(config_path)
    img_list = r"D:\Data\V1.2\v1.2\v1.2_0211\SS200000166BR_C1\test"
    files = glob.glob(os.path.join(img_list, '*.tif'))
    output_folder = r"D:\Data\V1.2\v1.2\v1.2_0211\SS200000166BR_C1\test_out"
    l = LineDetector(config_path, multiscale=False, ckpt_path=ckpt_path)
    l(files, output_path=None)
