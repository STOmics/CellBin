import cv2 as cv
import glog
import numpy as np
from numba import jit
import glob
from os.path import join, basename

# import matplotlib.pyplot as plt
# from kneed import KneeLocator
# from sklearn.linear_model import LinearRegression
#
# model = LinearRegression()

# from .shape import Line
# from shape import Line

# grid = 2
color_map = [(20, 1, 170), (255, 0, 255), (0, 255, 255)]


# def reject_outliers(data, m, direction):
#     all_data = data
#     if direction == "x":
#         data = data[:, 0]
#     else:
#         data = data[:, 1]
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     # if mdev > 2 or mdev == 0:
#     if mdev == 0:
#         return []
#     else:
#         s = d / mdev if mdev else 0.
#         keep = np.where(s < m)
#         return all_data[keep]
#
#
# @jit
# def find_cross(direction, mat):
#     pts = list()
#     h, w = mat.shape
#     counter = (direction == 'x' and h or w)
#     for i in range(0, counter, grid):
#         t = i + grid / 2
#         if direction == 'x':
#             region_mat = mat[i: i + grid, :w]
#             if region_mat.shape[0] != grid:
#                 continue
#             line = np.sum(region_mat, axis=0) / grid
#         else:
#             region_mat = mat[:h, i: i + grid]
#             if region_mat.shape[1] != grid:
#                 continue
#             line = np.sum(region_mat, axis=1) / grid
#         # p = argrelextrema(line, np.less_equal, order=100)
#         p = np.argmin(line)
#         if direction == 'x':
#             pts.append([p, t])
#         else:
#             pts.append([t, p])
#     return np.array(pts, dtype="float32")
#
#
# def point_to_line(pts, direction, h, w):
#     if len(pts) <= 2:
#         return -1, -1
#     if direction == "x":
#         x_train = pts[:, 1].reshape(-1, 1)
#         y_train = pts[:, 0]
#         model.fit(x_train, y_train)
#         y_predict = model.predict(x_train)
#     else:
#         x_train = pts[:, 0].reshape(-1, 1)
#         y_train = pts[:, 1]
#         model.fit(x_train, y_train)
#         y_predict = model.predict(x_train)
#     best_index = np.argmin(abs(y_predict - y_train))
#     best_x = x_train[best_index][0]
#     best_y = y_predict[best_index]
#     line = Line()
#     line.init_by_point_k((best_x, best_y), model.coef_[0])
#     if direction == "x":
#         p0, p1 = line.two_points((h, w))
#         p0, p1 = (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0]))
#     else:
#         p0, p1 = line.two_points((h, w))
#     return p0, p1
#
#
# def line(p1, p2):
#     A = (p1[1] - p2[1])
#     B = (p2[0] - p1[0])
#     C = (p1[0] * p2[1] - p2[0] * p1[1])
#     return A, B, -C
#
#
# def intersection(L1, L2):
#     D = L1[0] * L2[1] - L1[1] * L2[0]
#     Dx = L1[2] * L2[1] - L1[1] * L2[2]
#     Dy = L1[0] * L2[2] - L1[2] * L2[0]
#     if D != 0:
#         x = Dx / D
#         y = Dy / D
#         return int(x), int(y)
#     else:
#         return False


# def adjust(image):
#     # x = np.sum(image, axis=0)
#     h, w = image.shape
#     # poly_x = np.polyfit(range(len(x)), x, 2)
#     pts_x0, pts_x1, pts_y0, pts_y1 = -1, -1, -1, -1
#     # if poly_x[0] > 0:
#     pts_x = find_cross("x", image)
#     pts_x = reject_outliers(pts_x, m=2., direction="x")
#     if len(pts_x) != 0:
#         pts_x0, pts_x1 = point_to_line(pts_x, "x", h, w)
#     # y = np.sum(image, axis=1)
#     # poly_y = np.polyfit(range(len(y)), y, 2)
#     # if poly_y[0] > 0:
#     pts_y = find_cross("y", image)
#     pts_y = reject_outliers(pts_y, m=2., direction="y")
#     if len(pts_y) != 0:
#         pts_y0, pts_y1 = point_to_line(pts_y, "y", h, w)
#     if pts_x0 != -1 and pts_x1 != -1 and pts_y0 != -1 and pts_y1 != -1:
#         L1 = line(pts_x0, pts_x1)
#         L2 = line(pts_y0, pts_y1)
#         R = intersection(L1, L2)
#         return int(R[0]), int(R[1])
#     else:
#         return None, None


class PreProcess(object):
    def __init__(self):
        self.bins = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.tissue_thr = 0.0
        self.thread_pool = list()

    def set_depth(self, it):
        data_type = type(it)
        if data_type is np.uint8:
            self.depth = 8
        elif data_type is np.uint16:
            self.depth = 16

    def encode_range(self, arr, mode='median'):
        assert mode in ['median', 'hist', 'gmm', 'tissue_thr', 'none']
        data = arr.ravel()
        min_v = np.min(data)
        max_v = min_v + 10
        data_ = data[np.where(data <= self.tissue_thr)]

        if mode == 'median':
            var_ = np.std(data_)
            thr = np.median(data_)
            max_v = thr + var_
        elif mode == 'hist':
            freq_count, bins = np.histogram(data_, range(min_v, int(self.tissue_thr + 1)))
            count = np.sum(freq_count)
            freq = freq_count / count
            # kneedle_cov_inc = KneeLocator(bins[1:], freq, curve='convex', direction='decreasing', online=True)
            thr = bins[np.argmax(freq)]
            max_v = thr + (thr - min_v)
            # max_v = kneedle_cov_inc.knee
        elif mode == 'tissue_thr':
            max_v = self.tissue_thr
        else:
            max_v = np.max(data)
        return min_v, max_v

    def process(self, arr, multichannel):
        if arr.ndim == 3:
            self.height, self.width, _ = arr.shape
            self.set_depth(arr[0, 0, 0])
        else:
            self.height, self.width = arr.shape
            self.set_depth(arr[0, 0])
        self.tissue_thr = int((1 << self.depth) * (1 - 0.618))
        mat = np.zeros((self.height, self.width), dtype=np.uint8)
        try:
            min_v, max_v = self.encode_range(arr, mode='hist')
            encode(self.height, self.width, arr, max_v, min_v, mat)
        except:
            print("zou median le")
            min_v, max_v = self.encode_range(arr, mode='median')
            encode(self.height, self.width, arr, max_v, min_v, mat)
        if multichannel:
            mat = cv.cvtColor(mat, cv.COLOR_GRAY2BGR)
        return mat


@jit(nopython=True)
def encode(h, w, arr, max_v, min_v, mat):
    v_w = max_v - min_v
    if v_w == 0:
        v_w = 0.00001
    for r in range(h):
        for c in range(w):
            if arr[r, c] < min_v:
                mat[r, c] = 0
            elif arr[r, c] > max_v:
                mat[r, c] = 255
            else:
                mat[r, c] = int((arr[r, c] - min_v) * 255 / v_w)


class PostProcess(object):
    def __init__(self):
        self.image = None
        self.boxes = None

    def cross_points(self, image, boxes):
        pts = list()
        for b in boxes:
            # lt_x = box[0].lt()[0]
            # lt_y = box[0].lt()[1]
            # rb_x = box[0].rb()[0]
            # rb_y = box[0].rb()[1]
            # x, y = box[0].center()
            # cut_image = image[lt_y: rb_y, lt_x: rb_x]
            # adjust_x, adjust_y = adjust(cut_image)
            # if adjust_x is None or adjust_y is None:
            #     adjust_x, adjust_y = x, y
            # else:
            #     adjust_x, adjust_y = adjust_x + lt_x, adjust_y + lt_y
            # p = b[0].center()
            # p = (adjust_x, adjust_y)
            # pts.append([p, box[1]])
            p = b[0].center()
            pts.append([p, b[1]])
        return pts


class BBox(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.score = 0.0
        self.label = 0
        self.track_point = None

    def set(self, x, y, w, h, score, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.label = label

    def rb(self, ):
        return (self.x + self.w, self.y + self.h)

    def lt(self, ):
        return (self.x, self.y)

    def area(self, ):
        return self.w * self.h

    def center(self, ):
        return (self.x + int(self.w / 2), self.y + int(self.h / 2))


class CrossPointDetector(object):
    def __init__(self, width, height, confidence_thr, nms_thr):
        self.input_width = width
        self.input_height = height
        self.detector = None
        self.net = None
        self.boxes = None
        self.cross_points = None
        self.pre_process = PreProcess()
        self.post_process = PostProcess()
        self.confidence_thr = confidence_thr
        self.nms_thr = nms_thr
        self.data = None
        self.image = None

    def load_model(self, net_path, wights_path, GPU=False):
        net = cv.dnn.readNet(wights_path, net_path)
        if GPU:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
        else:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.detector = cv.dnn_DetectionModel(net)
        self.detector.setInputParams(size=(self.input_width, self.input_height), scale=1 / 255, swapRB=True)

    def inference(self, image):
        self.image = image
        arr = self.pre_process.process(image, multichannel=True)
        self.data = arr
        classes, scores, boxes = self.detector.detect(arr, self.confidence_thr, self.nms_thr)
        self.boxes = list()
        for (cls, score, box) in zip(classes, scores, boxes):
            x0, y0, w, h = box
            b = BBox()
            try: b.set(x0, y0, w, h, score[0], cls[0])
            except: b.set(x0, y0, w, h, score, cls)
            if b.x == 0 or b.y == 0 or b.rb()[0] >= image.shape[1] or b.rb()[1] >= image.shape[0]:
                continue
            if type(score) == np.float32: self.boxes.append([b, score])
            else: self.boxes.append([b, score[0]])
        self.cross_points = self.post_process.cross_points(self.image, self.boxes)

    def display(self, save_path):
        image = self.data
        if image.ndim != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        for i, box in enumerate(self.boxes):
            b, s = box
            x, y = b.center()
            cv.circle(image, b.center(), 1, (0, 255, 0), 1)
            lt_x = box[0].lt()[0]
            lt_y = box[0].lt()[1]
            rb_x = box[0].rb()[0]
            rb_y = box[0].rb()[1]
            cut_image = self.image[lt_y: rb_y, lt_x: rb_x]
            # cv.imwrite(fr"D:\Data\qc_issue\DP8400026176BL_B2\cut\{i}_{b.center()}.tif",
            #            cut_image)
            # adjust_x, adjust_y = adjust(cut_image)
            # if adjust_x is None or adjust_y is None:
            #     adjust_x, adjust_y = x, y
            # else:
            #     adjust_x, adjust_y = adjust_x + lt_x, adjust_y + lt_y
            # cv.circle(image, (adjust_x, adjust_y), 1, (0, 0, 255), 1)
            # cv.circle(image, (adjust_x, adjust_y), 10, (0, 0, 255), 1)
            # cv.rectangle(image, b.lt(), b.rb(), color_map[b.label], 10)
        # for p, score in self.cross_points:
        #     cv.circle(image, p, 1, (0, 255, 0), 1)
        #     t_p = []
        #     t_p.append(p[0] - 170)
        #     t_p.append(p[1])
        #     # cv.putText(image, str(round(score, 2)), tuple(t_p), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)
        #     # cv.putText(image, str([b.x, b.y]), tuple(t_p), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)
        cv.imwrite(save_path, image)
        return len(self.boxes)

    def _save(self, img_name, save_path):
        self._json()
        import json
        from copy import deepcopy
        cv.imwrite(save_path, self.data)
        cross_point = []
        for box in self.boxes:
            cross_label_template = deepcopy(self.cross_label)
            pts = []
            b, s = box
            pts.append([int(b.lt()[0]), int(b.lt()[1])])
            pts.append([int(b.rb()[0]), int(b.rb()[1])])
            cross_label_template["points"] = pts
            cross_point.append(cross_label_template)
        self.dc["shapes"] = cross_point
        self.dc["imagePath"] = img_name
        json_object = json.dumps(self.dc, indent=4)
        with open(save_path.replace(".tif", ".json"),
                  "w") as outfile:
            outfile.write(json_object)
        return len(self.boxes)

    def _json(self, ):
        self.cross_label = {
            "label": "cross",
            "points": None,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        self.dc = {"version": "0.0.2",
                   "flags": {},
                   "shapes": None,
                   "imagePath": None,
                   "imageData": None,
                   "imageHeight": None,
                   "imageWidth": None
                   }


def _load_model(img_shape, net_path="../../models/ST_TP.cfg",
                wights_path="../../models/ST_TP_v1.0.8_2000_11.29.weights"):
    ratio = round(img_shape[1] / img_shape[0], 1)
    net_height = 1024
    net_width = int(np.ceil(1024 / 256 * ratio) * 256)
    cpd = CrossPointDetector(net_width, net_height)
    cpd.load_model(net_path, wights_path)
    return cpd


def main(test_data, result=None, img_name=None):
    import os
    net_path = "../../models/ST_TP.cfg"
    wights_path_1_0_9 = "../../models/ST_TP_v1.0.9_1000_4_8.weights"
    save_1_0_9 = os.path.join(os.path.dirname(test_data), "yolo_v1.0.9_test_1")
    if not os.path.exists(save_1_0_9):
        os.mkdir(save_1_0_9)
    files = glob.glob(join(test_data, '*.tif'))
    arr = cv.imread(files[0], -1)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    ratio = round(arr.shape[1] / arr.shape[0], 1)
    net_height = 1024
    net_width = int(np.ceil(1024 / 256 * ratio) * 256)
    cpd_1_0_9 = CrossPointDetector(net_width, net_height, 0.5, 0.01)
    cpd_1_0_9.load_model(net_path, wights_path_1_0_9)
    lines = ''
    glog.info(f"working on v1.0.9")
    for f in files:
        lines += '{}\n\n'.format(basename(f))
        # f = join(test_data, '0003_0003_2021-07-07_12-40-37-278.tif')
        glog.info('Inference {}.'.format(basename(f)))
        arr = cv.imread(f, -1)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        cpd_1_0_9.inference(arr)
        if len(cpd_1_0_9.cross_points) == 0:
            continue
        for cp in cpd_1_0_9.cross_points:
            lines += '{} {}\n'.format(cp[0], cp[1])
        lines += '\n'
        c = f.split("\\")[-1]
        t = os.path.join(save_1_0_9, c)
        cpd_1_0_9.display(save_path=t)


if __name__ == '__main__':
    test_data = r"D:\Data\qc_issue\DP8400026176BL_B2\M1"
    main(test_data, result=None, img_name=None)
