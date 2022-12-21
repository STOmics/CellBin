import numpy as np
import glog
import pandas as pd
import os
import gzip


def export_roi_gene(file_path: str, roi: list):
    """
    Args:
        file_path: path of gene file
        roi: [x0, y0, w, h]
    Returns: None
    """

    if file_path.endswith('.gz'):
        f = gzip.open(file_path, 'rb')
    else: f = open(file_path, 'rb')

    glog.info('Start parse head info of file <{}>'.format(file_path))
    header = ''
    num_of_header_lines = 0
    eoh = 0
    for i, l in enumerate(f):
        l = l.decode("utf-8")  # read in as binary, decode first
        if l.startswith('#'):  # header lines always start with '#'
            header += l
            num_of_header_lines += 1
            eoh = f.tell()  # get end-of-header position
        else: break
    # find start of expression matrix
    f.seek(eoh)

    x0, y0, w, h = roi
    typeColumn = {
                    "geneID": 'str',
                    "x": np.uint32,
                    "y": np.uint32,
                    "values": np.uint32,
                    "UMICount": np.uint32,
                    "MIDCount": np.uint32,
                    "MIDCounts": np.uint32
                    }

    glog.info("Loading matrix data...")
    df = pd.read_csv(file_path, header=num_of_header_lines, sep='\t', dtype=typeColumn)

    if "UMICount" in df.columns: df = df.rename(columns={'UMICount':'MIDCount'})
    if "MIDCounts" in df.columns: df = df.rename(columns={'MIDCounts':'MIDCount'})

    df = df.drop(df[(df['x'] < x0) | (df['x'] > (x0 + w))].index)
    df = df.drop(df[(df['y'] < y0) | (df['y'] > (y0 + h))].index)
    df['x'] -= df['x'].min()
    df['y'] -= df['y'].min()
    output = os.path.join(os.path.dirname(file_path), 'SS2000.gem')
    glog.info('Write ROI gene file to {}'.format(output))
    # df.to_pickle(output, compression='gzip')

    df.to_csv(output, sep='\t', index=False)
    # with gzip.open(output.replace('.gem', '.gem.gz'), 'wb') as fd:
    #     fd.write(df)


def main():
    file_path = r'D:\code\mine\github\SS200000213BR_C5.gem.gz'
    roi = [12030, 16167, 5820, 4380]
    export_roi_gene(file_path, roi)


if __name__ == '__main__': main()
