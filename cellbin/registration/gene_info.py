import math
import os.path as osp
from sys import prefix
import pandas as pd
import numpy as np
import tifffile
import glog


__bit8_range__ = 256
__bit16_range__ = 65535


class GeneInfo(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    def __init__(self, gene_file, chip_mode=None):
        self.chip_mode = chip_mode
        self.orignal_point = None
        self.gene_mat = None
        self.coord = None
        self.total_exp_count = None
        self._load_gene_file(gene_file)
        #TODO: Check whether the chip sequence input is correct or not.
        #TODO: Check the legitimacy of documents.
    
    def batch_local_exp(self, anchor):
        indx, indy = anchor
        ind = np.where((self.coord[:, 2] == indx) & (self.coord[:, 3] == indy))
        return self.coord[ind][:, :2]

    def _load_gene_file(self, gene_file):
        """ Load the text file into memory and generate an image according to the spatial coordinate relationship """
        if not osp.exists(gene_file): raise IOError('File not found in specified path: {}.'.format(gene_file))
        suffix = osp.splitext(gene_file)[1]
        if suffix in ['.txt', '.tsv', '.gem', '.gz']:
            glog.info('Load Gene-Sequencing-Data from {}.'.format(gene_file)) 
            if suffix == '.gz': 
                import gzip
                fd = gzip.open(gene_file, 'rb')
            else: fd = open(gene_file, 'rb')

            # parse file head
            head_info = ''
            eoh = 0
            for line in fd:
                line = line.decode("utf-8")
                head_info += line
                if line.startswith('#'): eoh = fd.tell()
                else: break
            fd.seek(eoh)
            glog.info('File header information include:\n{}'.format(head_info.strip('\n')))
            # parse file body
            df = pd.read_csv(fd, sep='\t', header=0)
            # self.orignal_point = (df['x'].min(), df['y'].min())
            glog.info('The position where the Gene-Expression appears first is [XY] ({}, {}).'.format(df['x'].min(), df['y'].min()))
            df['x'] = df['x'] - df['x'].min()
            df['y'] = df['y'] - df['y'].min()
            max_x = df['x'].max() + 1
            max_y = df['y'].max() + 1
            glog.info('Size of the Gene-Expression region is [WxH] ({} x {}).'.format(max_x, max_y))
            ''' [UMICount, MIDCount] '''
            umi_count_name = df.columns.values.tolist()[-1]
            try: 
                new_df = df.groupby(['x', 'y']).agg(UMI_sum=(umi_count_name, 'sum')).reset_index()
                max_v = df[umi_count_name].max()
                self.total_exp_count = df[umi_count_name].sum()
            except: raise AttributeError('Columns have no attribute named as UMICount or MIDCount.')
            # df[umi_count_name].mean(), df[umi_count_name].median()
            glog.info('Gene-Expression count is [Maximum] ({}).'.format(max_v))
            # to image mat
            self.gene_mat_shape = (int(max_y), int(max_x))
            if max_v < __bit8_range__: self.gene_mat = np.zeros(shape=(max_y, max_x), dtype=np.uint8)
            else: self.gene_mat = np.zeros(shape=(max_y, max_x), dtype=np.uint16)
            self.gene_mat[new_df['y'], new_df['x']] = new_df['UMI_sum']
            glog.info('The Gene-Expression Map generate completedly.') 
        elif suffix == '.tif': 
            self.gene_mat = tifffile.imread(gene_file)
            glog.info('Completedly load Gene-Expression Map from {}.'.format(gene_file)) 
        else: raise IOError('{} format is an unsupported file type.'.format(suffix))
        if self.chip_mode: self.detect_grid()
    
    def dump_mat(self, save_path):
        suffix = osp.splitext(save_path)[1]
        if suffix in ['.tif', '.png', '.jpg']: 
            tifffile.imwrite(save_path, self.gene_mat, compress=True)
            glog.info('Dump Gene-Expression Map to {}.'.format(save_path))
        else: raise IOError('{} format is an unsupported file type.'.format(suffix))
    
    def dump_grid(self, save_path, filter=None):
        if self.chip_mode is None: raise IOError('The sequence to which the chip belongs is not specified.')
        suffix = osp.splitext(save_path)[1]
        if suffix in ['.txt']: 
            if not filter:
                np.savetxt(save_path, self.coord)
            else:
                x, y = filter
                coords = self.coord[np.where((self.coord[:, 2] == x) & (self.coord[:, 3] == y))]
                np.savetxt(save_path, coords)
            glog.info('Dump track points info to {}.'.format(save_path))
        else: raise IOError('{} format is an unsupported file type.'.format(suffix))

    def detect_grid(self, ):
        """ Locate the anchor position coordinate information in the stereo-seq matrix """
        def cross_points(x_intercept, y_intercept):
            cross_points_ = list()
            for x_ in x_intercept:
                for y_ in y_intercept:
                    x, ind_x = x_
                    y, ind_y = y_
                    if x < 0 or y < 0: continue
                    cross_points_.append([x, y, ind_x, ind_y])
            return np.array(cross_points_)

        assert self.chip_mode is not None
        x0, x1, y0, y1 = self._mass_center_region()
        glog.info('Inference track lines from [XY] ({}: {}, {}: {}).'.format(x0, x1, y0, y1))
        xgrid = self._single_direction_grid([x0, x1], axis=0)
        ygrid = self._single_direction_grid([y0, y1], axis=1)
        self.coord = cross_points(xgrid, ygrid)
        glog.info('Got track points by the XY lines completely.')

    def _mass_center_region(self, mode=0):
        """ Preliminarily defined as the circumscribed rectangular frame of the organization area """
        def line_center(line):
            xx = np.array(range(len(line)))
            xx_cal = xx * line
            return np.sum(xx_cal) / np.sum(line)

        def x0x1(xc, fov, wmax):
            x0 = int(xc) - fov // 2
            x1 = x0 + fov
            if x0 < 0: 
                x0 = 0
                x1 = x0 + fov
            if x1 >= wmax:
                x1 = wmax
                x0 = x1 - fov
            return [x0, x1]

        xline = np.sum(self.gene_mat, 0)
        yline = np.sum(self.gene_mat, 1)
        if mode == 0:
            x_mass = line_center(xline)
            y_mass = line_center(yline)
        elif mode == 1:
            x_mass = np.argmax(xline)
            y_mass = np.argmax(yline)
        mode_info = {0: 'Mass-Center', 1: 'MaxV-Center'}
        glog.info('By method {}, Center is ({}, {})'.format(mode_info[mode], x_mass, y_mass))
        h, w = self.gene_mat_shape
        fovw, fovh = np.sum(self.chip_mode, axis=1)
        x0, x1 = x0x1(x_mass, fovw, wmax=w) 
        y0, y1 = x0x1(y_mass, fovh, wmax=h) 
        return [x0, x1, y0, y1]
    
    def _single_direction_grid(self, region, axis=0, line_w=3):
        """ Locate the track line position information in a specific direction """
        def get_intercept(intercept, region, ind, templ):
            count = len(templ)
            idx = intercept
            intercept = [[idx, ind]]
            s, e = region
            # face to large
            while idx < e:
                ind = ind % count
                item_len = templ[ind]
                idx += item_len
                intercept.append([idx, (ind + 1) % count])
                ind += 1
            # face to small
            idx, ind = intercept[0]
            while idx > s:
                ind -= 1
                ind = ind % count
                item_len = templ[ind]
                idx -= item_len
                intercept.append([idx, ind])
            return sorted(intercept, key=(lambda x: x[0]))

        start, end = region
        if axis == 0: roi = self.gene_mat[:, start: end]
        else: roi = self.gene_mat[start: end, :]
        index = np.where(np.sum(roi, axis=axis)==0)[0]
        diff = np.diff(index)
        start += index[np.argmax(diff[:line_w]) - 1]
        diff = diff[:(len(diff) // line_w) * line_w]
        diff = np.reshape(diff, (len(diff) // line_w, line_w))
        match = np.max(diff, axis=1) + (line_w - 1)
        template_used = self.chip_mode[axis] * (math.ceil(len(match) / len(self.chip_mode[axis])) + 1)
        std_var = list()
        for i in range(len(self.chip_mode[axis])):
            items = template_used[i: i + len(match)]
            std_var.append(np.std([(match[i] / items[i]) for i in range(len(match))]))
        ind = np.argmin(std_var) % len(self.chip_mode[axis]) 
        fitune = line_w / 2 - line_w // 2
        return get_intercept(start + fitune, (0, self.gene_mat_shape[axis]), ind, self.chip_mode[axis])


def chunked_file_reader(file, block_size=1024 * 1024 * 256):
    """ Reduce memory consumption of very large matrix files """
    fd = open(file, 'r')
    head_info = ''
    eoh = 0
    for line in fd:
        # line = line.decode("utf-8")
        head_info += line
        
        if line.startswith('#'): eoh = fd.tell()
        else: 
            eoh += len(line)
            break
    fd.seek(eoh)
    """生成器函数：分块读取文件内容，使用 iter 函数
    """
    # 首先使用 partial(fp.read, block_size) 构造一个新的无需参数的函数
    # 循环将不断返回 fp.read(block_size) 调用结果，直到其为 '' 时终止
    # for chunk in iter(partial(fd.read, block_size), ''):
    #     yield chunk
    i = 0
    suffix = ''
    prefix = ''
    while True:
        p = fd.read(block_size)
        i += 1

        if p: 
            e = p[-30:].split('\n')[-1]
            suffix_len = len(e)
            if suffix_len: buffer = prefix + p[:-suffix_len]
            else: buffer = prefix + p
            # v = np.fromstring(buffer, dtype=('|S10', int, int, int), sep='\n')
            v = np.char.split(buffer, sep='\n')
            print(v.shape)
            # np.loadtxt
            # print('\n {}=>'.format(i))
            # print(buffer[:50], '\n##\n', buffer[-50:])
            # if i == 7: print(p[:300], '#', p[-300:])
            # print('\n {} ** {}'.format(p[:50], p[-50:]))
            prefix = '\n{}'.format(e)
            yield p
        else: 
            fd.close()
            return


if __name__ == '__main__':
    file_path = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/liuhuanlin/02.data/shenjuan/SS200000763BR_A5.gem'
    chip = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
            [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    # Only need tif.
    # gf = GeneInfo(gene_file=file_path)
    # Need tif and track-points info
    import time
    t0 = time.time()

    for p in chunked_file_reader(file_path):
        pass
    # chunked_file_reader(file_path, block_size=1024 * 8)
    print('start: {}'.format(time.time() - t0))

    # gf = GeneInfo(gene_file=file_path, chip_mode=chip)
    # gf.dump_mat(save_path='/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/liuhuanlin/02.data/shenjuan/SS200000763BR_A5.tif')
    # gf.dump_grid(save_path='/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/liuhuanlin/02.data/shenjuan/SS200000763BR_A5.txt')

    

