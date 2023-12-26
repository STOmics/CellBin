# Cell Segmentation
**1. run**

python segment.py -i /path/to/image -o /path/to/save/result -g gpu_id/cpu

    -i image path
    -o save result path
    -g value in [-1, 0, 1, 2, ...], -1: use cpu, 0: use gpu 0, 1: use gpu 1, ...


**2. Example**

> python segment.py -i img.tif -o result/ -g -1

> python segment.py -i img/ -o result/ -g 0
