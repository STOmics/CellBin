import numpy as np
from cellbin import CellBinElement
from cellbin.registration.alignment import AlignByTrack


class Registration(CellBinElement):
    def __init__(self):
        super(Registration, self).__init__()
        self.align_track = AlignByTrack()

        self.offset = [0, 0]
        self.rot90 = 0
        self.flip = False
        self.score = 0

        self.regist_img = np.array([])
        self.fov_transformed = np.array([])
        self.dist_shape = None

    def mass_registration_trans(
            self,
            fov_transformed,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            fov_stitched_shape,
            rotation,
            flip

    ):
        self.align_track.set_chip_template(chip_template=chip_template)
        self.fov_transformed = fov_transformed
        self.dist_shape = vision_image.shape
        adjusted_stitch_template = self.align_track.adjust_cross(
            stitch_template=track_template,
            scale_x=scale_x,
            scale_y=scale_y,
            fov_stitched_shape=fov_stitched_shape,
            new_shape=self.fov_transformed.shape,
            chip_template=chip_template,
            rotation=rotation
        )
        vision_cp = self.align_track.find_track_on_vision_image(vision_image, chip_template)

        offset, rot_type, score = self.align_track.run(
            transformed_image=self.fov_transformed,
            vision_img=vision_image,
            vision_cp=vision_cp,
            stitch_tc=adjusted_stitch_template,
            flip=flip,
        )

        # result update
        self.flip = flip
        self.rot90 = rot_type
        self.offset = offset
        self.score = score

        return 0

    def mass_registration_stitch(
            self,
            fov_stitched,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            rotation,
            flip

    ):
        # transform
        self.fov_transformed = self.stitch_to_transform(
            fov_stitch=fov_stitched,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=rotation
        )
        fov_stitched_shape = fov_stitched.shape
        self.mass_registration_trans(
            self.fov_transformed,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            fov_stitched_shape,
            rotation,
            flip
        )

        return 0

    @staticmethod
    def stitch_to_transform(fov_stitch, scale_x, scale_y, rotation):
        from cellbin import ImageTransform
        i_trans = ImageTransform()
        i_trans.set_image(fov_stitch)
        fov_transformed = i_trans.rot_scale(
            x_scale=scale_x,
            y_scale=scale_y,
            angle=rotation
        )
        return fov_transformed

    def transform_to_regist(self, flip=True):
        from cellbin import ImageTransform
        i_trans = ImageTransform()
        i_trans.set_image(self.fov_transformed)
        if flip:
            i_trans.flip(
                flip_type='hor'
            )
        i_trans.rot90(self.rot90)
        self.regist_img = i_trans.offset(self.offset[0], self.offset[1], self.dist_shape)


if __name__ == '__main__':
    import json
    from glob import glob
    import os
    import tifffile
    import cv2

    regist_path = r"D:\Data\regist\FP200000340BR_D1"
    json_path = glob(os.path.join(regist_path, "7_result", "**.json"))[0]
    with open(json_path, "r") as f:
        json_obj = json.load(f)
        scale_x = json_obj["RegisterInfo"]["Scale"]
        scale_y = json_obj["RegisterInfo"]["Scale"]
        rotation = json_obj["RegisterInfo"]["Rotation"]
        chip_template = json_obj["ChipInfo"]["FOVTrackTemplate"]
        offset_ori = json_obj["AnalysisInfo"]["input_dct"]["offset"]
        rot_ori = json_obj["AnalysisInfo"]["input_dct"]["rot_type"]
    # fov_transformed_path = os.path.join(regist_path, '4_register', 'fov_stitched_transformed.tif')
    # fov_transformed = tifffile.imread(fov_transformed_path)
    fov_stitched_path = os.path.join(regist_path, '2_stitch', 'fov_stitched.tif')
    fov_stitched = tifffile.imread(fov_stitched_path)

    # czi mouse brain -> stitch shape (2, x, x)
    if len(fov_stitched.shape) == 3:
        fov_stitched = fov_stitched[0, :, :]

    try:
        gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**raw.tif"))[0]
    except IndexError:
        try:
            gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**_gene_exp.tif"))[0]
        except IndexError:
            gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**.gem.tif"))[0]
    gene_exp = cv2.imread(gene_exp_path, -1)

    track_template = np.loadtxt(os.path.join(regist_path, "2_stitch", "template.txt"))  # stitch template
    flip = True
    im_shape = np.loadtxt(os.path.join(regist_path, "4_register", "im_shape.txt"))
    rg = Registration()
    rg.mass_registration_stitch(
        fov_stitched,
        gene_exp,
        chip_template,
        track_template,
        scale_x,
        scale_y,
        rotation,
        flip
    )
    print(rg.offset, rg.rot90, rg.score)
    rg.transform_to_regist()
    regist_img = rg.regist_img
    print("asd")
    tifffile.imwrite(r"D:\Data\regist\FP200000340BR_D1\test_regist.tif", regist_img)


if __name__ == '__main__':
    import json
    from glob import glob
    import os
    import tifffile
    import cv2

    regist_path = r"D:\Data\regist\FP200000340BR_D1"
    json_path = glob(os.path.join(regist_path, "7_result", "**.json"))[0]
    with open(json_path, "r") as f:
        json_obj = json.load(f)
        scale_x = json_obj["RegisterInfo"]["Scale"]
        scale_y = json_obj["RegisterInfo"]["Scale"]
        rotation = json_obj["RegisterInfo"]["Rotation"]
        chip_template = json_obj["ChipInfo"]["FOVTrackTemplate"]
        offset_ori = json_obj["AnalysisInfo"]["input_dct"]["offset"]
        rot_ori = json_obj["AnalysisInfo"]["input_dct"]["rot_type"]
    # fov_transformed_path = os.path.join(regist_path, '4_register', 'fov_stitched_transformed.tif')
    # fov_transformed = tifffile.imread(fov_transformed_path)
    fov_stitched_path = os.path.join(regist_path, '2_stitch', 'fov_stitched.tif')
    fov_stitched = tifffile.imread(fov_stitched_path)

    # czi mouse brain -> stitch shape (2, x, x)
    if len(fov_stitched.shape) == 3:
        fov_stitched = fov_stitched[0, :, :]

    try:
        gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**raw.tif"))[0]
    except IndexError:
        try:
            gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**_gene_exp.tif"))[0]
        except IndexError:
            gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**.gem.tif"))[0]
    gene_exp = cv2.imread(gene_exp_path, -1)

    track_template = np.loadtxt(os.path.join(regist_path, "2_stitch", "template.txt"))  # stitch template
    flip = True
    im_shape = np.loadtxt(os.path.join(regist_path, "4_register", "im_shape.txt"))
    rg = Registration()
    rg.mass_registration_stitch(
        fov_stitched,
        gene_exp,
        chip_template,
        track_template,
        scale_x,
        scale_y,
        rotation,
        flip
    )
    print(rg.offset, rg.rot90, rg.score)
