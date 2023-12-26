import os
import numpy as np
import pandas as pd
import glog
import os.path as osp
import cv2
import onnxruntime as ort
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageQualityAssessment(object):
    def __init__(self, si, info):
        self.info = info
        self.si = si
        self.cut_size = 512
        self.overlap = 0
        self.img_size = 256
        self.GPU = False
        self.chip_name = self.si.QCInfo.OutputName
        self.out_path = self.si.QCInfo.OutputPath
        self.image_score = None
        self.result_df = None
        curr_dir = osp.dirname(osp.abspath(__file__))
        model_dir = osp.join(curr_dir, 'IQA_models')
        self.model_path = os.path.join(model_dir, 'IQA_v1.1.onnx')
        self.sess = self.model_load()
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def split(self, image):
        shape = image.shape
        x_nums = int(shape[0] / (self.cut_size - self.overlap))
        y_nums = int(shape[1] / (self.cut_size - self.overlap))
        data = []
        for x_temp in range(x_nums + 1):
            for y_temp in range(y_nums + 1):
                x_begin = max(0, x_temp * (self.cut_size - self.overlap))
                y_begin = max(0, y_temp * (self.cut_size - self.overlap))
                x_end = min(x_begin + self.cut_size, shape[0])
                y_end = min(y_begin + self.cut_size, shape[1])
                if x_begin == x_end or y_begin == y_end:
                    continue
                i = image[x_begin: x_end, y_begin: y_end]
                i = cv2.resize(i, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                data.append(i)
        return data

    def get_data(self, X):
        X = np.array(X)
        return self.preprocess(X)

    def preprocess(self, X):
        X = X / 255.
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        return X

    def bytescaling(self, data, cmin=None, cmax=None, high=255, low=0):
        if data.dtype == np.uint8:
            return data
        if high > 255:
            high = 255
        if low < 0:
            low = 0
        if high < low:
            raise ValueError("`high` should be greater than or equal to `low`.")
        if cmin is None:
            cmin = np.min(data)
        if cmax is None:
            cmax = np.max(data)
        cscale = int(cmax) - int(cmin)
        if cscale == 0:
            cscale = 1
        scale = float(high - low) / cscale
        bytedata = (data - cmin) * scale + low
        return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

    def model_load(self):
        ep_list = ['CPUExecutionProvider']
        glog.info(f"currently using model {self.model_path}")
        sess = ort.InferenceSession(self.model_path, providers=ep_list)
        return sess

    def csv_generate(self, result):
        self.result_df = pd.DataFrame(result, columns=["img_name", "score"])
        df_save_path = os.path.join(self.out_path, f"{self.chip_name}.csv")
        self.result_df.to_csv(df_save_path, index=False)
        return df_save_path

    def plot_result(self,):
        shape = (self.si.ImageInfo.ScanRows, self.si.ImageInfo.ScanCols)
        data = np.ones(shape)
        data = data * -1
        for idx, row in self.result_df.iterrows():
            score = row["score"]
            row, col = row["img_name"]
            data[row][col] = score
        sns.heatmap(data, cmap="Reds", annot=True)
        plot_save_path = os.path.join(self.out_path, f"{self.chip_name}.png")
        plt.xlabel("# of Row")
        plt.ylabel("# of Column")
        plt.title("Tissue Section QC Result")
        plt.savefig(f"{plot_save_path}")
        plt.clf()  # clear the plot
        return plot_save_path

    def predict(self):
        score_result = []
        g_total = 0
        b_total = 0
        for i in tqdm(range(self.info.fov_rows)):
            for j in range(self.info.fov_cols):
                arr = self.info.get_image(i, j)
                # todo:缺失的fov是否发出警告么
                if arr is None:
                    continue
                data = self.split(arr)
                if len(data) == 0:
                    continue
                X_test = self.get_data(data)
                X_test = X_test.astype(np.float32)  # v1.2 remove
                y_pred = self.sess.run([self.output_name], {self.input_name: X_test})[0].argmax(-1)
                black = (y_pred == 2).sum()
                total = len(y_pred)
                if total - black == 0:
                    continue
                good = (y_pred == 1).sum()
                g_total += good
                b_total += (total - black - good)
                fov_score = good / (total - black)
                score_result.append([(i, j), fov_score])
        if (g_total + b_total) == 0:
            self.image_score = 0
        else:
            self.image_score = g_total / (g_total + b_total)
        glog.info(f"The score for whole image is: {self.image_score}")
        df_save_path = self.csv_generate(score_result)
        plot_save_path = self.plot_result()
        glog.info("IQA finished!")
        return self.image_score, df_save_path, plot_save_path


if __name__ == '__main__':
    json_path = r"D:\Work\Data\test_classification_model\SS200000170BR_B6\1_origin\SS200000170BR_B6_20211231.json"
    input_path = r"D:\Work\Data\test_classification_model\SS200000170BR_B6\1_origin\B6"
    fov_name = "SS200000170BR_B6"
    out_path = r"D:\Work\Data\test_classification_model\SS200000170BR_B6\1_origin\tttt"
