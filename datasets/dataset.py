import argparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt

class ImgAugment(object):
    def __init__(self, width = 512, height = 512, phase='train'):
        super(ImgAugment, self).__init__()
        self.phase = phase
        self.width = width
        self.height = height
        self.transform_train = iaa.Sequential(
            [
                iaa.HorizontalFlip(),
            ]
        )
        self.transform_test = iaa.Sequential(
            [
                iaa.size.Resize(size=(width, height))
            ]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            ])
    def __call__(self, image, bbs):
        img = np.asarray(image)
        if self.phase == 'train':
            img, bbs = self.transform_train(image=img, bounding_boxes=bbs)

        img_aug, bbs_aug = self.transform_test(image=img, bounding_boxes=bbs)
        img_aug = self.transform(img_aug)
        return img_aug, bbs_aug


class WheetDataset(Dataset):

    def __init__(self, df, root='./data', phase='train', height=1024, width=1024):
        super(Dataset, self).__init__()
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.root = root
        self.phase = phase
        self.transforms = ImgAugment(phase=phase, height=height, width=width)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'train', self.image_ids[idx]+'.jpg')
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        records = self.df[self.df['image_id']==self.image_ids[idx]]

        boxes = []
        bbs = []
        num_objs = len(records)
        for i in range(num_objs):
            x1, y1, width, height = [float(corr) for corr in records.iloc[i, 3][1:-1].split(',')]
            x2 = x1 + width
            y2 = y1 + height
            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        bbs_oi = BoundingBoxesOnImage(bbs, shape=img.shape)
        img, bbs = self.transforms(img, bbs_oi)
        # debug = False
        # if debug:
        #     image_after = bbs.draw_on_image(img, size=2)
        #     plt.imshow(image_after)
        #     plt.show()
        boxes = bbs.to_xyxy_array()
        boxes = torch.stack([torch.from_numpy(box) for box in boxes]).float()

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.image_ids)
