import tifffile
import numpy as np
import cv2 as cv
import glog
import pandas as pd


class CellMatrixLoader(object):
    def __init__(self, ) -> None:
        self.matrix_path = None
        self.cell_data = None
        self.cell_coord = None
        self.cell_exp = None

    def load(self, matrix_path, cell_mask_path):
        self.matrix_path = matrix_path
        typeColumn = {
            "geneID": 'str',
            "x": np.uint32,
            "y": np.uint32,
            "values": np.uint32,
            "UMICount": np.uint32,
            "MIDCount": np.uint32,
            "MIDCounts": np.uint32
        }

        header = self._parse_matrix_head()
        glog.info("Loading matrix data...")
        genedf = pd.read_csv(matrix_path, header=header, sep='\t', dtype=typeColumn)
        x0, y0, x1, y1 = [genedf['x'].min(), genedf['y'].min(), genedf['x'].max(), genedf['y'].max()]
        glog.info('Matrix original point is ({}, {}), shape is {}'.format(x0, y0, (y1 - y0 + 1, x1 - x0 + 1)))

        if "UMICount" in genedf.columns: genedf = genedf.rename(columns={'UMICount': 'MIDCount'})
        if "MIDCounts" in genedf.columns: genedf = genedf.rename(columns={'MIDCounts': 'MIDCount'})

        tissuedf = pd.DataFrame()
        glog.info('Loading cell mask...')
        cell_mask = tifffile.imread(cell_mask_path)

        x, y = np.where(cell_mask > 0)
        min_ = min(np.min(x), np.min(y))
        max_ = max(np.max(x), np.max(y))
        mask_crop = cell_mask[min_:max_ + 1, min_:max_ + 1]

        glog.info('Mask shape, Matrix shape {}, ({}, {})'.format(cell_mask.shape, y1 - y0 + 1, x1 - x0 + 1))
        assert ((y1 - y0 + 1 == cell_mask.shape[0]) and (x1 - x0 + 1 == cell_mask.shape[1]))
        _, markers = cv.connectedComponents(mask_crop, connectivity=4)

        img = np.zeros((cell_mask.shape[0], cell_mask.shape[1]), dtype=np.int32)
        img[min_: max_ + 1, min_: max_ + 1] += markers
        markers = img

        glog.info('Cell mask shape is {}, cell count is {}'.format(cell_mask.shape, _ - 1))
        dst = np.nonzero(markers)

        glog.info("Create single cell gene data...")
        tissuedf['x'] = dst[1] + genedf['x'].min()
        tissuedf['y'] = dst[0] + genedf['y'].min()
        tissuedf['CellID'] = markers[dst]

        # keep background data
        self.cell_exp = pd.merge(genedf, tissuedf, on=['x', 'y'], how='left').fillna(0)
        del cell_mask
        del tissuedf
        del dst
        del markers
        del genedf

        if 'MIDCounts' in self.cell_exp.columns:
            self.cell_exp = self.cell_exp.rename(columns={'MIDCounts': 'UMICount'})
        if 'MIDCount' in self.cell_exp.columns:
            self.cell_exp = self.cell_exp.rename(columns={'MIDCount': 'UMICount'})

        for it in ['UMICount', 'x', 'y', 'geneID']: assert it in self.cell_exp.columns

        self.cell_data = self.cell_exp[self.cell_exp.CellID != 0].copy()
        self.cell_coord = self.cell_data.groupby('CellID').mean()[['x', 'y']].reset_index()

    def _parse_matrix_head(self, ):
        if self.matrix_path.endswith('.gz'):
            import gzip
            f = gzip.open(self.matrix_path, 'rb')
        else:
            f = open(self.matrix_path, 'rb')

        glog.info('Start parse head info of file <{}>'.format(self.matrix_path))

        header = ''
        num_of_header_lines = 0
        eoh = 0
        for i, l in enumerate(f):
            l = l.decode("utf-8")  # read in as binary, decode first
            if l.startswith('#'):  # header lines always start with '#'
                header += l
                num_of_header_lines += 1
                eoh = f.tell()  # get end-of-header position
            else:
                break
        # find start of expression matrix
        f.seek(eoh)
        return num_of_header_lines


class CellCorrect(object):
    def __init__(self, ) -> None:
        self.loader = CellMatrixLoader()
        self.output_path = ''

    def set_output(self, p): self.output_path = p

    def adjust(self, ): pass

    def creat_cell_gxp(self, matrx_path, cell_mask_path):
        self.loader.load(matrx_path, cell_mask_path)

    def tissue_bin(self, tissue_mask: str, bin_size: int): pass

    def cell_bin(self, tissue_mask=None):
        if tissue_mask is not None: pass

    def visualization(self, export_path: str): pass


# def main():
#     matrix_path = r'D:\data\gene\SS200000135TL_D1.gem.gz'
#     cell_mask_path = r'D:\data\stereoCell\SS200000135TL_D1_mask.tif'
#     cml = CellMatrixLoader()
#     cml.load(matrix_path, cell_mask_path)
#
#
# if __name__ == '__main__':
#     main()
