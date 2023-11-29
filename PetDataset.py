# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import os
import cv2
from torch.utils.data import Dataset


class PetDataset(Dataset):
    def __init__(self, img_dir, training=True, transform=None):

        self.img_dir = img_dir
        self.training = training

        if self.training:
            self.mode = 'train'
            self.label_file = 'train_noses.2.txt'
        else:
            self.mode = 'test'
            self.label_file = 'test_noses.txt'

        self.transform = transform
        self.num = 0
        self.img_files = []

        f = open(self.label_file)
        for line in f.readlines():
            img_file = line.split(',')[0]
            self.img_files += [img_file]

        self.max = len(self)

    def prepare(self, image, label):
        return self.transform()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = self.transform(cv2.imread(img_path, cv2.IMREAD_COLOR))

        label = []
        f = open(self.label_file)
        for line in f.readlines():
            if self.img_files[idx] in line:
                split_string = line.split(',')
                x = int(split_string[1][2:])
                y = int(split_string[2][1:-3])
                label = [x,y]

        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)

