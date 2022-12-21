import glog
import pandas as pd
from sklearn.mixture import GaussianMixture
import os
from rich.progress import track
import math
import tqdm
from multiprocessing import Pool
import numpy as np
try:
    from .cell_correct import CellCorrect
except:
    from cell_correct import CellCorrect


class GMMCorrect(CellCorrect):
    def __init__(self) -> None:
        super().__init__()
        self.process = 10
        self.radius = 50
        self.threshold = 20
        self.correct_data = None

    def set_radius(self, r): self.radius = r

    def set_threshold(self, thr): self.threshold = thr

    def set_process(self, p): self.process = p

    def _func(self, x, p_num):
        p_data = []
        if not os.path.exists(os.path.join(self.output_path, 'bg_adjust_label')):
            os.mkdir(os.path.join(self.output_path, 'bg_adjust_label'))
        for i in tqdm.tqdm(x, desc='proc {}'.format(p_num)):
            try:
                clf = GaussianMixture(n_components=3, covariance_type='spherical')

                cell_test = self.loader.cell_exp[
                    (self.loader.cell_exp.x < self.loader.cell_coord.loc[i].x + self.radius) &
                    (self.loader.cell_exp.x > self.loader.cell_coord.loc[i].x - self.radius) &
                    (self.loader.cell_exp.y > self.loader.cell_coord.loc[i].y - self.radius) &
                    (self.loader.cell_exp.y < self.loader.cell_coord.loc[i].y + self.radius)
                ]

                # fit GaussianMixture Model
                clf.fit(cell_test[cell_test.CellID == self.loader.cell_coord.loc[i].CellID][['x', 'y', 'UMICount']].values)
                cell_test_bg_ori = cell_test[cell_test.CellID == 0]
                bg_group = cell_test_bg_ori.groupby(['x', 'y']).agg(UMI_max=('UMICount', 'max')).reset_index()
                cell_test_bg = pd.merge(cell_test_bg_ori, bg_group, on=['x', 'y'])		
                # threshold 20

                score = pd.Series(-clf.score_samples(cell_test_bg[['x', 'y', 'UMI_max']].values))
                cell_test_bg['score'] = score
                threshold = self.threshold
                cell_test_bg['CellID'] = np.where(score < threshold, self.loader.cell_coord.loc[i].CellID, 0)

                # used multiprocessing have to save result to file
                p_data.append(cell_test_bg)
            except Exception as e:
                print(e)
                with open(os.path.join(self.output_path, 'error_log.txt'), 'a+') as f:
                    f.write('Cell ID: {}\n'.format(self.loader.cell_coord.loc[i].CellID))

        out = pd.concat(p_data)
        out.drop('UMI_max', axis=1, inplace=True)
        out.to_csv(os.path.join(self.output_path, 'bg_adjust_label', '{}.txt'.format(p_num)), sep='\t', index=False)

    def _score(self, ):
        qs = math.ceil(len(self.loader.cell_coord.index) / int(self.process))
        pool = Pool(processes=self.process)
        for i in range(self.process):
            idx = np.arange(i * qs, min((i + 1) * qs, len(self.loader.cell_coord.index)))
            if len(idx) == 0: continue
            pool.apply_async(self._func, args=(idx, i))
        pool.close()
        pool.join()
        return None

    def _correction(self, ):
        bg_data = []
        error = []

        file_list = os.listdir(os.path.join(self.output_path, 'bg_adjust_label'))
        for i in track(file_list, description='Correction'):
            try:
                tmp = pd.read_csv(os.path.join(self.output_path, 'bg_adjust_label', i), sep='\t')
                bg_data.append(tmp[tmp.CellID != 0])
            except: error.append(i)
        adjust_data = pd.concat(bg_data).sort_values('score')
        adjust_data = adjust_data.drop_duplicates(subset=['geneID', 'x', 'y', 'UMICount'],
                                                  keep='first').rename(columns={'score': 'tag'})
        adjust_data['tag'] = 1
        self.loader.cell_data['tag'] = 0
        self.correct_data = pd.concat([adjust_data, self.loader.cell_data])

    def export(self, ):
        self.correct_data = \
            self.correct_data.astype({"geneID": str, "x": int, "y": int, "UMICount": int, "CellID": int, "tag": int})
        self.correct_data.to_csv(os.path.join(self.output_path, 'GMM_data_adjust.txt'), sep='\t', index=False)

    def adjust(self, ):
        glog.info('Start adjust...')
        self._score()
        self._correction()
        self.export()
