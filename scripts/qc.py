import glob
import shutil
import h5py
import glog
import sys
import getpass
import os
import argparse


def image_qc(image_path, output_path, stereo_chip):
    """ ImageQC that generate anchor points for registration """
    glog.info('Anchor point positioning for subsequent registration process')
    cmd = '{} {} -i {} -o {} -c {} -n {} -e {}'.format(
        sys.executable, '../cellbin/iqc/qc_util.py', image_path, output_path, stereo_chip, 'test', getpass.getuser())
    print(cmd)
    os.system(cmd)


def main(args, para):

    fdir = os.path.dirname(os.path.abspath(__file__))
    tmp = os.path.join(fdir, '..', 'tmp')
    if not os.path.exists(tmp): os.makedirs(tmp)

    input = args.tiles_path
    output = tmp
    chip_no = args.chip_no

    image_qc(image_path=input, output_path=output, stereo_chip=chip_no)

    ipr = glob.glob(os.path.join(tmp, '*.ipr'))[0]
    h5 = h5py.File(ipr, 'r')
    qc_flag = h5['QCInfo'].attrs['QCPassFlag']
    if qc_flag == 1: glog.info('Image QC: PASS')
    else: glog.warn('Image QC: FAIL')
    h5.close()
    shutil.rmtree(tmp)


""" Usage
python .\qc.py --tiles_path D:\data\test\SS200000135TL_D1 --chip_no SS200000135TL_D1
"""
if __name__ == '__main__':
    usage=""" StereoCell """
    PROG_VERSION='v0.0.1'

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-t", "--tiles_path", action="store", dest="tiles_path", type=str, required=True, help="The path of tile images.")
    parser.add_argument("-c", "--chip_no", action="store", dest="chip_no", type=str, required=True, help="Stereo-seq chip No.")
    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
