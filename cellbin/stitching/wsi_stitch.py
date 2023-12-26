import numpy as np
from cellbin.utils.util import rc_key
from cellbin import Image


class StitchingWSI(object):
    """ Stitching WSI """
    def __init__(self):

        """Initialize Stitching WSI.

        Args:
          fov_rows: The number of FOV scans in the Y direction
          fov_cols: The number of FOV scans in the X direction
          fov_height: Height of FOV image.
          fov_width: Width of FOV image.
          fov_channel: The image channel count of mosaic image.
          fov_dtype: Dtype of pixel.
          fov_location: Each FOV space coordinate matrix..
          buffer: Anchor Image of stitching
          mosaic_height: Height of mosaic image
          mosaic_width: Width of mosaic image
        """
        self.fov_rows = None
        self.fov_cols = None
        self.fov_height = self.fov_width = 0
        self.fov_channel = self.fov_dtype = 0
        self.fov_location = None
        self._overlap = 0.1
        self.buffer = None
        self.mosaic_width = self.mosaic_height = None

    def set_overlap(self, overlap): self._overlap = overlap

    def _init_parm(self, src_image: dict):
        test_image_path = list(src_image.values())[0]
        img = Image()
        img.read(test_image_path)

        self.fov_height = img.height
        self.fov_width = img.width
        self.fov_channel = img.channel
        self.fov_dtype = img.dtype

    def _set_location(self, loc):
        if loc is not None:
            h, w = loc.shape[:2]
            assert (h == self.fov_rows and w == self.fov_cols)
            self.fov_location = loc
        else:
            self.fov_location = np.zeros((self.fov_rows, self.fov_cols, 2), dtype=int)
            for i in range(self.fov_rows):
                for j in range(self.fov_cols):
                    self.fov_location[i, j] = [
                        int(i * self.fov_height * (1 - self._overlap)),
                        int(j * self.fov_width * (1 - self._overlap))]
        x0 = np.min(self.fov_location[:, :, 0])
        y0 = np.min(self.fov_location[:, :, 1])
        self.fov_location[:, :, 0] -= x0
        self.fov_location[:, :, 1] -= y0
        x1 = np.max(self.fov_location[:, :, 0])
        y1 = np.max(self.fov_location[:, :, 1])
        self.mosaic_width, self.mosaic_height = [x1 + self.fov_width, y1 + self.fov_height]

    def mosaic(self, src_image: dict, loc=None, downsample=1, multi=False):
        """ Stitching image for FOVs """
        rc = np.array([k.split('_') for k in list(src_image.keys())], dtype=int)

        self.fov_rows, self.fov_cols = loc.shape[:2]
        self._init_parm(src_image)

        self._set_location(loc)
        img = Image()
        h, w = (int(self.mosaic_height * downsample), int(self.mosaic_width * downsample))
        if self.fov_channel == 1: self.buffer = np.zeros((h, w), dtype=self.fov_dtype)
        else: self.buffer = np.zeros((h, w, self.fov_channel), dtype=self.fov_dtype)

        if multi:
            pass
        else:
            import tqdm
            for i in tqdm.tqdm(range(self.fov_rows), desc='FOVs Stitching'):
                for j in range(self.fov_cols):
                    img.read(src_image[rc_key(i, j)])
                    arr = img.image
                    x, y = self.fov_location[i, j]
                    x_, y_ = (int(x/downsample), int(y/downsample))
                    if self.fov_channel == 1:
                        self.buffer[y_: y_ + int(self.fov_height // downsample), x_: x_ + int(self.fov_width // downsample)] = \
                            arr[::downsample, ::downsample]
                    else:
                        self.buffer[y_: y_ + int(self.fov_height // downsample), x_: x_ + int(self.fov_width // downsample), :] = \
                            arr[::downsample, ::downsample, :]

    def save(self, output_path, compression=False):
        img = Image()
        img.image = self.buffer
        img.write(output_path, compression=compression)


def main():
    src_image = {}
    wsi = StitchingWSI()
    wsi.set_overlap(0.1)
    wsi.mosaic(src_image)


if __name__ == '__main__': main()
