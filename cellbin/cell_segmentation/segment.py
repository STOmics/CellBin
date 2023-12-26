#!/usr/bin/env python
#coding: utf-8
###### Import Modules ##########
import argparse
import glog
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waterSeg'))
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postProcess'))
# print(sys.path)
import time

import seg_utils.cell_seg_pipeline as pipeline


#########################################################
#########################################################
##input:    tif image after register
##pipeline: 1.image preprocess
##          2.model inference
##          3.image postprocess
##return:   mask in [0,1]
#########################################################
#########################################################


usage = '''
     time:  %s
     Usage: %s imagePath outPath is_gpu(cpu:-1; gpu:id)
''' % ('2022-10-10', os.path.basename(sys.argv[0]))


def args_parse():
    ap = argparse.ArgumentParser(usage=usage)
    ap.add_argument('-i', '--img_path', action='store', help='image path')
    ap.add_argument('-o', '--out_path', action='store',  help='mask path')
    # ap.add_argument('-w', '--Watershed', default=0, action='store', type=int, help='watershed')
    ap.add_argument('-g', '--GPU', action='store', help='cpu:-1;gpu:id')

    return ap.parse_args()


def cell_seg(img_path, out_path, flag=1, DEEP_CROP_SIZE=20000, OVERLAP = 100):
    cell_seg_pipeline = pipeline.CellSegPipe(img_path, out_path, flag, DEEP_CROP_SIZE, OVERLAP)
    cell_seg_pipeline.run_cell_seg()


def segment_entry(args, verbose=''):
    """ Config cell segmentation & inference """
    args = vars(args)
    img_path = args['img_path']
    out_path = args['out_path']
    flag = 1  # args['Watershed']
    gpu = args['GPU']
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    t0 = time.time()
    cell_seg(img_path, out_path, flag, DEEP_CROP_SIZE=256)  # DEEP_CROP_SIZE=5000
    t1 = time.time()
    glog.info('total time: %.2f' % (t1 - t0))


def main():
    ######################### Phrase parameters #########################
    args = args_parse()   
    glog.info(args)

    # call segmentation
    segment_entry(args)


if __name__ == '__main__':
    main()
