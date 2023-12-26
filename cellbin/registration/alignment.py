from scipy.spatial.distance import cdist


def rotate(ptx, pty, angle, original_shape, new_shape):
    """ Rotate the track point """
    px, py = ptx, pty
    ori_h, ori_w = original_shape
    new_h, new_w = new_shape
    cx = ori_w / 2
    cy = ori_h / 2
    rad = math.radians(angle)
    new_px = cx + (px - cx) * math.cos(rad) + (py - cy) * math.sin(rad)
    new_py = cy + -((px - cx) * math.sin(rad)) + (py - cy) * math.cos(rad)
    x_offset, y_offset = (new_w - ori_w) / 2, (new_h - ori_h) / 2
    new_px += x_offset
    new_py += y_offset
    return new_px, new_py


def f_padding(img, top, bot, left, right, mode='constant', value=0):
    """
    update by dengzhonghan on 2023/2/23
    1. support 3d array padding.
    2. not support 1d array padding.

    Args:
        img (): numpy ndarray (2D or 3D).
        top (): number of values padded to the top direction.
        bot (): number of values padded to the bottom direction.
        left (): number of values padded to the left direction.
        right (): number of values padded to the right direction.
        mode (): padding mode in numpy, default is constant.
        value (): constant value when using constant mode, default is 0.

    Returns:
        pad_img: padded image.

    """

    if mode == 'constant':
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode, constant_values=value)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode, constant_values=value)
    else:
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode)
    return pad_img


class AlignByTrack:
    """ Align By Track """

    def __init__(self):
        """Initialize AlignByTrack.
        Vision image: generated based on gene expression matrix
        Transformed image obtained based on
            - stitched image
            - scale and rotation

        Transformed image should be in the same scale compared to vision image

        Transformed image and vision image only contain:
            - n times 90 degree rotation (e.g. 90, 180, 270, etc.)
            - x, y direction offsets

        Args:
            self.x_template (list): chip template on x direction
            self.y_template (list): chip template on y direction
            self.fov_size (float): length of a period on chip template
            self.dist_thresh (float): maximum distance threshold
            self.transformed_shape (tuple): shape of tranformed image
            self.transformed_mass (nd array): mass center of transformed image
            self.vision_shape (tuple): shape of vision image
            self.vision_mass (nd array): mass center of vision image
            self.vision_img (nd array): vision image
            self.transformed_image (nd array): transformed image
            self.transformed_cfov_pts (nd array): selected cross points on transform image
            self.vision_cfov_pts (nd array): selected cross points on vision image
        """

        self.search_angle_set = (0, 90, 180, 270)
        self.search_range_x = [-2, -1, 0, 1, 2]
        self.search_range_y = [-2, -1, 0, 1, 2]

        self.x_template = None
        self.y_template = None
        self.fov_size = None
        self.dist_thresh = None
        self.transformed_shape = None
        self.transformed_mass = None
        self.vision_shape = None
        self.vision_mass = None
        self.vision_img = None
        self.transformed_image = None
        self.transformed_cfov_pts = None
        self.vision_cfov_pts = None

    def set_chip_template(self, chip_template):
        self.x_template = chip_template[0]
        self.y_template = chip_template[1]
        self.fov_size = np.sum(self.x_template)
        self.dist_thresh = np.sum(self.x_template) * 2 / 3

    @staticmethod
    def find_track_on_vision_image(vision_img, template):
        vision_cross_pts = find_cross(vision_img, template)
        return vision_cross_pts

    @staticmethod
    def adjust_cross(stitch_template, scale_x, scale_y, fov_stitched_shape, new_shape, chip_template, rotation, flip=True):
        scale_shape = np.array([fov_stitched_shape[0] * scale_y, fov_stitched_shape[1] * scale_x])

        stitch_template[:, 0] = stitch_template[:, 0] * scale_x
        stitch_template[:, 1] = stitch_template[:, 1] * scale_y

        pts = stitch_template[:, :2]
        ids = stitch_template[:, 2:]

        new_px, new_py = rotate(
            pts[:, 0:1],
            pts[:, 1:2],
            rotation,
            original_shape=scale_shape,
            new_shape=new_shape,
        )

        if flip:
            new_px = new_shape[1] - 1 - new_px  # 翻转, 注意这里是用的图片的形状, 所以要-1
            chip_xlen, chip_ylen = [len(chip_template[0]), len(chip_template[1])]
            ids[:, 0] = chip_xlen - 1 - ids[:, 0]
        pts_ids = np.hstack((new_px, new_py, ids))
        return pts_ids

    @staticmethod
    def cal_score(transformed_image, vision_image, offset):
        if offset[0] < 0:
            left_x = int(round(abs(offset[0])))
            vision_image = f_padding(vision_image, 0, 0, left_x, 0)
        else:
            vision_image = vision_image[:, int(round(offset[0])):]

        if offset[1] < 0:
            up_y = int(round(abs(offset[1])))
            vision_image = f_padding(vision_image, up_y, 0, 0, 0)
        else:
            vision_image = vision_image[int(round(offset[1])):, :]

        shape_vision = np.shape(vision_image)
        shape_transform = np.shape(transformed_image)

        if shape_vision[0] > shape_transform[0]:
            vision_image = vision_image[:shape_transform[0], :]
        else:
            vision_image = f_padding(vision_image, 0, shape_transform[0] - shape_vision[0], 0, 0)

        if shape_vision[1] > shape_transform[1]:
            vision_image = vision_image[:, :shape_transform[1]]
        else:
            vision_image = f_padding(vision_image, 0, 0, 0, shape_transform[1] - shape_vision[1])
        score = np.sum(np.multiply(vision_image, transformed_image))

        return score

    @staticmethod
    def get_pts_based_on_ids(pts_ids, keep_ids=(4, 4)):
        keep = (pts_ids[:, 2] == keep_ids[0]) & (pts_ids[:, 3] == keep_ids[1])
        selected_pts = pts_ids[keep][:, :2]
        return selected_pts

    @staticmethod
    def down_sample_normalize(img):
        img = img[::5, ::5]
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 100
        return img

    @staticmethod
    def get_new_shape(old_shape, angle):
        angle = math.radians(angle)
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)
        h, w = old_shape
        new_w = round(h * math.fabs(angle_sin) + w * math.fabs(angle_cos))
        new_h = round(w * math.fabs(angle_sin) + h * math.fabs(angle_cos))
        new_shape = (new_h, new_w)
        return new_shape

    @staticmethod
    def get_rough_offset(offset_guess, rot_guess, old_shape, new_shape, transformed_pts, vision_pts, dist_thresh):
        rot_x, rot_y, = rotate(
            transformed_pts[:, 0:1],
            transformed_pts[:, 1:2],
            rot_guess,
            old_shape,
            new_shape
        )
        transformed_pts_temp = np.hstack((rot_x, rot_y))

        # get qualified pts
        dist = cdist(transformed_pts_temp + offset_guess, vision_pts)
        qualified = np.min(dist, axis=1) <= dist_thresh
        transformed_pts_qualified = transformed_pts_temp[qualified]
        dist_qualified = dist[qualified]
        vision_pt_qualified = vision_pts[np.argmin(dist_qualified, axis=1)]

        if len(transformed_pts_qualified) > 0:
            x_offset = -np.median(np.array(transformed_pts_qualified.T[0] - vision_pt_qualified.T[0]))
            y_offset = -np.median(np.array(transformed_pts_qualified.T[1] - vision_pt_qualified.T[1]))
        else:
            x_offset, y_offset = 0, 0
        return x_offset, y_offset

    def search_fov(self, offset_ori, angle):
        white_image = self.transformed_image
        white_image = np.rot90(white_image, angle // 90)
        vision_image = self.vision_img
        score_max = 0
        offset_last = []

        # 依次遍历上下左右9个FOV的匹配程度
        for row in self.search_range_x:
            for col in self.search_range_y:
                offset_temp = [offset_ori[0] + col * self.fov_size, offset_ori[1] + row * self.fov_size]
                score_temp = self.cal_score(white_image, vision_image, np.array(offset_temp) / 5)
                if score_temp > score_max:
                    score_max = score_temp
                    offset_last = offset_temp
        return np.array(offset_last), score_max

    def get_best_in_all_angles_offsets(self):
        score_record = []
        offset_record = []
        for num, angle in enumerate(self.search_angle_set):
            old_shape = self.transformed_shape
            new_shape = self.get_new_shape(old_shape, angle)
            rot_mass_x, rot_mass_y, = rotate(
                self.transformed_mass[0],
                self.transformed_mass[1],
                angle,
                self.transformed_shape,
                new_shape,
            )

            white_mass_temp = np.array([rot_mass_x, rot_mass_y])
            offset_temp = self.vision_mass - white_mass_temp
            rough_x_offset, rough_y_offset = self.get_rough_offset(
                offset_guess=offset_temp,
                rot_guess=angle,
                old_shape=old_shape,
                new_shape=new_shape,
                transformed_pts=self.transformed_cfov_pts,
                vision_pts=self.vision_cfov_pts,
                dist_thresh=self.dist_thresh
            )
            if rough_x_offset != 0 and rough_y_offset != 0:
                offset, score = self.search_fov(np.array([rough_x_offset, rough_y_offset]), angle)
            else:
                offset, score = [], 0
            offset_record.append(offset)
            score_record.append(score)

        rot_type = score_record.index(max(score_record))
        offset = offset_record[rot_type]

        return offset, rot_type, max(score_record)

    def run(self, transformed_image, vision_img, vision_cp, stitch_tc, flip):
        self.transformed_shape = transformed_image.shape
        self.transformed_mass = get_mass(transformed_image)
        self.vision_shape = vision_img.shape
        self.vision_mass = get_mass(vision_img)
        transformed_image = self.down_sample_normalize(transformed_image)
        self.vision_img = self.down_sample_normalize(vision_img)
        if flip:
            self.transformed_image = np.fliplr(transformed_image)
        else:
            self.transformed_image = transformed_image

        self.transformed_cfov_pts = self.get_pts_based_on_ids(stitch_tc)
        self.vision_cfov_pts = self.get_pts_based_on_ids(vision_cp)

        offset, rot_type, score = self.get_best_in_all_angles_offsets()
        return offset, rot_type, score


import numpy as np
import math


def get_mass(image):
    image = image.astype(float)
    image_x = np.sum(image, 0)
    xx = np.array(range(len(image_x)))
    xx_cal = xx * image_x
    x_mass = np.sum(xx_cal) / np.sum(image_x)

    image_y = np.sum(image, 1)
    yy = np.array(range(len(image_y)))
    yy_cal = yy * image_y
    y_mass = np.sum(yy_cal) / np.sum(image_y)
    return np.array([x_mass, y_mass])


def find_cross_ind(position, summation):
    result = []
    for _ in range(len(summation) - np.max(position) - 1):
        position += 1
        result.append(sum(summation[position]))
    first_ind = result.index(min(result)) + 2
    return first_ind


def find_first_tp(image, mass_center, chip_template, find_range=3000):
    min_x = int(round(mass_center[0]) - find_range)
    max_x = int(round(mass_center[0]) + find_range)
    min_y = int(round(mass_center[1]) - find_range)
    max_y = int(round(mass_center[1]) + find_range)
    find_image = image[min_y: max_y, min_x: max_x].astype(float)
    x_sum = np.sum(find_image, 0)  # vertical
    y_sum = np.sum(find_image, 1)  # horizontal
    position_x, position_y = np.cumsum(np.insert(np.array(chip_template), 0, 0, axis=1)[:, :-1], axis=1)
    x_first = find_cross_ind(position_x, x_sum)
    y_first = find_cross_ind(position_y, y_sum)
    x_first += min_x
    y_first += min_y
    return x_first, y_first


def one_to_all(template, mid_pos, length):
    step_size = sum(template)
    upper = math.ceil((length - mid_pos) / step_size)
    lower = math.ceil(mid_pos / step_size)
    interval = np.concatenate(
        (
            np.cumsum(np.tile(template, upper)),
            np.array([0]),
            np.cumsum(np.tile(-np.array(template[::-1]), lower))
        )
    )
    index = np.concatenate(
        (
            np.tile(np.arange(len(template)), lower),
            np.tile(np.arange(len(template)), upper),
            np.array([0])
        )
    )
    interval.sort()
    interval = mid_pos + interval
    interval = interval.reshape(-1, 1)
    index = index.reshape(-1, 1)
    combined = np.concatenate(
        (interval, index),
        axis=1
    )
    return combined


def find_cross(gene_exp, chip_template):
    mass_center = get_mass(gene_exp)
    x_mid, y_mid = find_first_tp(gene_exp, mass_center, chip_template)
    h, w = gene_exp.shape
    x_all = one_to_all(chip_template[0], x_mid, w)
    y_all = one_to_all(chip_template[1], y_mid, h)

    all_comb = np.concatenate(
        (
            np.tile(
                np.expand_dims(x_all, 1),
                (1, y_all.shape[0], 1)
            ),
            np.tile(
                np.expand_dims(y_all, 0),
                (x_all.shape[0], 1, 1)
            )
        ),
        axis=2
    ).reshape(-1, 4)

    all_comb[:, [1, 2]] = all_comb[:, [2, 1]]
    final_result = all_comb[
        np.logical_and.reduce(
            (
                all_comb[:, 0] <= w,
                all_comb[:, 1] <= h,
                all_comb[:, 0] >= 0,
                all_comb[:, 1] >= 0
            )
        )
    ]
    return final_result


if __name__ == '__main__':
    import cv2

    vision_image = cv2.imread(r"D:\Data\regist\FP200000340BR_D1\3_vision\FP200000340BR_D1_gene_exp.tif", -1)
    chip_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420], [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    new_cp = find_cross(vision_image, chip_template)
    print("asd")
