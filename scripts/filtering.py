import argparse

import numpy as np


def main(args, para):
    import tifffile
    a = tifffile.imread(args.tissue_mask)
    a[np.where(a > 1)] = 1
    b = tifffile.imread(args.nuclei_mask)
    b[np.where(b > 1)] = 1
    assert a.shape == b.shape
    tifffile.imwrite(args.output_file, a * b * 255, compression="zlib", compressionargs={"level": 8})


if __name__ == '__main__':
    usage = """Mask merge """
    PROG_VERSION = 'v0.0.1'

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-t", "--tissue_mask", action="store", dest="tissue_mask", type=str, required=True, help="Input tissue mask.")
    parser.add_argument("-n", "--nuclei_mask", action="store", dest="nuclei_mask", type=str, required=True, help="Input nuclei mask.")
    parser.add_argument("-o", "--output_file", action="store", dest="output_file", type=str, required=True,
                        help="Result output file.")
    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
