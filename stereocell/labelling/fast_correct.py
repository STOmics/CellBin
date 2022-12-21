import pandas as pd
from rich.progress import track
import numpy as np
import glog
try:
    from .cell_correct import CellCorrect
except:
    from cell_correct import CellCorrect


class FastCorrect(CellCorrect):
    def __init__(self) -> None:
        super().__init__()
        self.cell_label_list = None
        self.cell_points = None
        self.free_points_np = None
        self.data = None

    def adjust(self, ):
        glog.info('Start adjust...')
        self.loader.cell_exp['tag'] = 0
        self.data = self.loader.cell_exp.to_numpy()

        self.cell_points = {}
        free_points = []
        for i, row in enumerate(self.data):
            if row[4] != 0:
                if row[4] not in self.cell_points: self.cell_points[row[4]] = [[row[1], row[2]]]
                else: self.cell_points[row[4]].append([row[1], row[2]])
            else: free_points.append((row[1], row[2], i))

        free_points = sorted(list(set(free_points)), key=lambda x: x[0])
        self.free_points_np = np.asarray(free_points)
        self.cell_label_list = sorted(list(self.cell_points.keys()))
        self.allocate_free_pts()

        self.export()

    def allocate_free_pts(self, ):
        x_axis = self.free_points_np[:, 0]
        for cell in track(self.cell_label_list, description='correcting cells'):
            min_x, max_x = min(self.cell_points[cell], key=lambda x: x[0])[0], max(self.cell_points[cell], key=lambda x: x[0])[0]
            min_y, max_y = min(self.cell_points[cell], key=lambda x: x[1])[1], max(self.cell_points[cell], key=lambda x: x[1])[1]

            idx_x_low, idx_x_upper = self.find_nearest(x_axis, min_x - 10), self.find_nearest(x_axis, max_x + 10)
            sub_free_pts_np = self.free_points_np[idx_x_low:idx_x_upper, :2]
            sub_free_pts_idx = self.free_points_np[idx_x_low:idx_x_upper, 2]

            pts_np = np.array(self.cell_points[cell])
            x, y = np.sum(pts_np, axis=0)
            centroid = [int(x / len(pts_np)), int(y / len(pts_np))]

            length = max(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) // 2 + 3, 5)

            dis_matrix = self.distance(sub_free_pts_np, np.array(centroid))
            idx = np.where(dis_matrix < length)
            self.data[sub_free_pts_idx[idx], -2] = cell
            self.data[sub_free_pts_idx[idx], -1] = 1

    @staticmethod
    def find_nearest(array, value): return np.searchsorted(array, value, side="left")

    @staticmethod
    def distance(p1, p2): return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))

    def export(self, ):
        columns = list(self.loader.cell_exp.columns)
        test_df = pd.DataFrame(self.data, columns=columns)
        test_df = test_df.astype({"geneID": str, "x": int, "y": int, "UMICount": int, "CellID": int, "tag": int})
        import os
        test_df.to_csv(os.path.join(self.output_path, 'fast_data_adjust.txt'), sep='\t', index=False)
