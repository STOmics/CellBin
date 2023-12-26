import numpy as np
import os
import torch
from cellbin.cell_segmentation.seg_utils.utils import split, merge
from .resnet_unet import EpsaResUnet
from albumentations.pytorch import ToTensorV2
from albumentations import (HorizontalFlip, Normalize, Compose, GaussNoise)
from tqdm import tqdm
import cv2
import glog
from .dataset import data_batch
import tifffile


class CellInfer(object):
    """ Cell segmentation """

    def __init__(self, file, batch_size):
        """ Initialize Cell segmentation.

        Args:
            file: Stain image
            batch_size: batch size used during inference
        """
        self.file = file
        self.batch_size = batch_size

    def get_transforms(self):
        list_transforms = []

        list_transforms.extend(
            [
                # HorizontalFlip(p=0.5),
                # GaussNoise(p=0.7),
            ])
        list_transforms.extend(
            [
                # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                ToTensorV2(),
            ])
        list_trfms = Compose(list_transforms)
        return list_trfms

    def run_infer(self):
        """
        Complete the generation of cell segmentation results through preprocessing,
        splitting large images, reasoning, and sub-block merging

        Returns: cell mask
        """
        if isinstance(self.file, list):
            file_list = self.file
        else:
            file_list = [self.file]

        result = []

        model_path = os.path.join(os.path.split(__file__)[0], '../../weights')
        model_dir = os.path.join(model_path, 'stereocell_bcdu_cell_256x256_220926.pth')
        model = EpsaResUnet(out_channels=6)
        glog.info('Load model from: {}'.format(model_dir))
        model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        glog.info('Load model ok.')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): glog.info('GPU type is {}'.format(torch.cuda.get_device_name(0)))
        glog.info(f"using device: {device}")
        model.to(device)
        for idx, image in enumerate(file_list):
            label_list = []
            h, w = image.shape
            glog.info('image shape: {}'.format(image.shape))

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            img_list, x_list, y_list = split(image, 256, 100)

            # image padding to 256*256
            ori_size = []
            pad_img_list = np.full((len(img_list), 256, 256, 3), 0, dtype='uint8')
            for ind, i in enumerate(img_list):
                ori_size.append([i.shape[0], i.shape[1]])
                pad_img_list[ind, :i.shape[0], :i.shape[1], :] = i
            img_list = pad_img_list

            # total_num = len(img_list)
            glog.info('[image %d/%d]' % (idx + 1, len(file_list)))
            dataset = data_batch(img_list)
            test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
            for batch in tqdm(test_dataloader, ncols=80):
                img = batch
                img = img.type(torch.FloatTensor)
                img = img.to(device)

                pred_mask = model(img)
                bacth_size = len(pred_mask)
                pred_mask = torch.sigmoid(pred_mask).detach().cpu().numpy()
                pred1 = pred_mask[:, 0, :, :]
                pred1[:] = (pred1[:] > 0.35) * 255
                pred1 = pred1.astype(np.uint8)
                pred2 = pred_mask[:, 3, :, :]
                pred2[:] = (pred2[:] > 0.3) * 255
                pred2 = pred2.astype(np.uint8)
                pred = cv2.bitwise_or(pred1, pred2)
                for i in range(bacth_size):
                    label_list.append(pred[i, :, :])

            for ind, ii in enumerate(label_list):
                label_list[ind] = ii[:ori_size[ind][0], :ori_size[ind][1]]

            merge_label = merge(label_list, x_list, y_list, [h, w])
            result.append(merge_label)

        return result
