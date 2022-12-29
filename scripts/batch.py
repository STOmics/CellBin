import os
import glog


#  stitching
todo = [r'D:\data\StereoCellStitching\MIST\Phase_Image_Tiles\images',
        r'D:\data\StereoCellStitching\StereoCell\B01207D3F4\B01207D3F4',
        r'D:\data\StereoCellStitching\StereoCell\FP200004565_J7K8\FP200004565_J7K8',
        r'D:\data\StereoCellStitching\StereoCell\SS200000064_BADC\SS200000064_BADC',
        r'D:\data\StereoCellStitching\StereoCell\SS200000654_JAKA\SS200000654_JAKA',
        r'D:\data\StereoCellStitching\StereoCell\SS200000975BR_A6\SS200000975BR_A6',
        ]

glog.info('>> TEST for Stitching')

for image_path in todo:
    prefix = os.path.basename(image_path).split('.')[0]
    out_path = os.path.join(os.path.dirname(image_path), 'StereoCell')
    cmd = r'python ..\stereocell\stitching\stitch.py --input {} --output {}'.format(
        image_path, os.path.join(out_path, '{}.tif'.format(prefix)))
    os.system(cmd)
    break
