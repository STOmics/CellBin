# _*_ coding: utf-8 _*_
"""
Date:    2023/3/6 18:03
Author:  Qiang Kang
Email:   kangqiang@genomics.cn
"""

import argparse
import pandas as pd
import numpy as np
import tifffile as tiff


def Load_Gxp_Bin(geneExpFile):
    """ load gem file """
    gem = pd.read_csv(geneExpFile, sep="\t", quoting=3, comment="#")
    return gem


def Load_TissueCut(tcImaFile):
    """ Load tissue cut image """
    tci = tiff.imread(tcImaFile)
    return tci


def Tissue_Filter(df, tci):
    ds = []
    for ind in range(len(df)):
        xg, yg = df['x'].get(ind), df['y'].get(ind)
        if tci[yg, xg] == 0:
            ds.append(ind)
    ds = np.array(ds)
    df.drop(labels=ds, axis=0, inplace=True)
    return df


'''
python tissue_bin.py -g D:\data\test\SS200000135TL_D1.gem.gz -t D:\data\test\paper\registration_tissue_mask.tif -o D:\data\test\paper
'''
def args_parse():
    arg = argparse.ArgumentParser(description='BinX filtering')
    arg.add_argument('-g', '--gem', type=str, required=True, help='gem file')
    arg.add_argument('-t', '--tissuecut', type=str, required=False, help='tissue cut image')
    arg.add_argument('-o', '--outputpath', type=str, required=True, help='output path')
    args = arg.parse_args()
    return args


def main():
    args = args_parse()
    allowed_columns = ['geneID', 'x', 'y', 'MIDCount']
    gem = Load_Gxp_Bin(args.gem)
    gem_columns = gem.columns.to_list()
    if 'UMICount' in gem_columns:
        gem.rename(columns={'UMICount': 'MIDCount'}, inplace=True)
    assert set(allowed_columns) <= set(gem_columns), 'Error, the required column does not exist in this data'
    if len(allowed_columns) < len(gem_columns):
        gem = gem[allowed_columns]
    if args.tissuecut is not None:
        tci = Load_TissueCut(args.tissuecut)
        gem = Tissue_Filter(gem, tci)
    gem.to_csv(args.outputpath + '/bin_filtering.gem', sep='\t', index=False)
    print('BinX filtering finished')


if __name__ == "__main__":
    main()