"""Defines the fisheye dataset for directional marking point detection."""
import json
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from collections import namedtuple
import torch

SIZE_IMAGES=17164
POSE_NUM = 9
CalibrationPoints = namedtuple('CalibrationPoints', ('image2d_points', 'world3d_points'))


class CameraPoseDataset(Dataset):
    """fisheye dataset."""
    def __init__(self, root):
        super(CameraPoseDataset, self).__init__()
        self.root = root
        self.temp_names=[]# all pictures in BFLR in order
        self.file_name = []
        self.image_transform = ToTensor()
        for index in range(POSE_NUM):
            for image_index in range(SIZE_IMAGES):
                self.temp_names.append(str(index)+str(image_index).zfill(7))

    def __getitem__(self, index):
        mark = 'B'
        images = []
        images_gray = []
        image2d_points_set = []
        world3d_points_set = []
        for iter_index in range(4):
            if iter_index == 0:
                mark = 'B'
            if iter_index == 1:
                mark = 'F'
            if iter_index == 2:
                mark = 'L'
            if iter_index == 3:
                mark = 'R'
            image2d_points = []
            world3d_points = []
            image_name = self.temp_names[index]
            image = cv.imread(self.root+mark+'/'+image_name+'.jpg')
            image_clone = image.copy()
            image_gray = cv.cvtColor(image_clone,cv.COLOR_BGR2GRAY)
            images_gray.append(self.image_transform(image_gray))
            image = cv.resize(image, (512, 512))
            images.append(self.image_transform(image))
            with open(self.root+'C_f_labels/label'+ image_name[0] + '_'+mark+'.json') as file:
                for point in json.load(file)['world3d_points']:
                    world3d_points.append(point)
            with open(self.root + 'C_f_labels/label' + image_name[0] + '_' +mark+ '.json') as file:
                for point in json.load(file)['image2d_points']:
                    image2d_points.append(point)
            image2d_points=torch.Tensor(image2d_points)
            world3d_points=torch.Tensor(world3d_points)
            image2d_points_set.append(image2d_points)
            world3d_points_set.append(world3d_points)
        all_images = torch.stack([images[0],images[1],images[2],images[3]])
        all_gray_images = torch.stack([images_gray[0],images_gray[1],images_gray[2],images_gray[3]])
        all_2d_points = torch.stack([image2d_points_set[0],image2d_points_set[1],image2d_points_set[2],image2d_points_set[3]])
        all_3d_points = torch.stack([world3d_points_set[0],world3d_points_set[1],world3d_points_set[2],world3d_points_set[3]])
        return all_images,all_2d_points,all_3d_points,all_gray_images

    def __len__(self):
        return len(self.temp_names)

