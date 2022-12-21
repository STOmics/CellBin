import os
import glog


def main():
    image_path = r'D:\code\mine\github\StereoCell\data\SS2000_regist.tif'
    matrix_path = r'D:\code\mine\github\StereoCell\data\SS2000.gem.gz'
    out_path = r'D:\code\mine\github\StereoCell\data'

    # test pipeline
    # glog.info('>> TEST for Pipeline')
    # cmd = r'python ..\stereocell\cell_bin.py --image_path {} --matrix_path {} --out_path {}'.format(image_path,
    #                                                                                     matrix_path,
    #                                                                                     out_path)
    # os.system(cmd)
    prefix = os.path.basename(image_path).split('.')[0]

    # test tissue segmentation
    glog.info('>> TEST for Tissue Segmentation')
    cmd = r'python ..\stereocell\segmentation\tissue.py --input {} --output {}'.format(
        image_path, os.path.join(out_path, '{}_tissue_mask.tif'.format(prefix)))
    os.system(cmd)

    # test cell segmentation
    glog.info('>> TEST for Cell Segmentation')
    cell = os.path.join(out_path, '{}_cell_mask.tif'.format(prefix))
    cmd = r'python ..\stereocell\segmentation\cell.py --input {} --output {}'.format(
        image_path, cell)
    os.system(cmd)

    # test matrix correct
    glog.info('>> TEST for Matrix Correct')
    cmd = r'python ..\stereocell\labelling\correct.py --mask_path {} --matrix_path {} --out_path {}'.format(cell, matrix_path, out_path)
    os.system(cmd)


if __name__ == '__main__':
    main()
