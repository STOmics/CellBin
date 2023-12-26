import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fov_stitching'))
import fov_info
import argparse
import glog
from fov_qc import QualityControl

###### Version and Date
PROG_VERSION = '0.0.2'
PROG_DATE = '2021-08-01'

###### Usage
USAGE = """

     Version %s by lhl %s

     Usage: python %s <json_path> >STDOUT
""" % (PROG_VERSION, PROG_DATE, os.path.basename(sys.argv[0]))


def qc_entry(opt, args):
    info = fov_info.FOVInfo(args[0], opt.image_path)
    if opt.output is None:
        info.set_output_path(os.path.split(args[0])[0])
    else:
        info.set_output_path(opt.output)
    qc = QualityControl(info)
    qc.qc()
    # sth.update_json(info.scope_info, args[0])


def main():
    # test_stitch_entry('D:\\DATA\\stitchingv2_test\\motic\\result\\demo\\scope_info.json')
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="Save path of QC result files.")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="FOV images storage location.")
    (opt, args) = arg_parser.parse_known_args()
    if len(args) != 1:
        arg_parser.print_help()
        glog.error('The parameters number is not correct.')
        return 1

    qc_entry(opt, args)

    return 0


if __name__ == '__main__':
    return_code = main()
    sys.exit(return_code)
