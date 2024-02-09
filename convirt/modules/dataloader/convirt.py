import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLRDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, csv_file, root_dir, transform=None, clip=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images/texts.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clr_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.clip = clip

    def __len__(self):
        return len(self.clr_frame)

    def text_sampling(self, text):
        text = text.replace("\n", " ")
        # if self.sampling:
        text = text.split(".")
        if '' in text:
            text.remove('')
        text = random.choice(text)
        return text

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.clr_frame.iloc[idx, 0]
                                )
        if not os.path.exists(img_name):
            img_name=img_name.replace('train','validation')
        image = Image.open(img_name)
        if not self.clip:
            image = image.convert('RGB')
        text = self.clr_frame.iloc[idx, 1]
        cuis = self.clr_frame.iloc[idx, 2]
        name = self.clr_frame.iloc[idx, 0]
        sample = {'image': image, 'text': text,'name':name}
        # sample = {'image': image, 'text': text, 'cuis':cuis,'name':name}
        if self.clip:
            sample = self.transform(sample['image']), sample['text'],sample['name'] #self.text_sampling(sample['text'])
        elif self.transform:
            sample = self.transform(sample)

        return sample

class IRMADataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, csv_file, root_dir, mode, transform=None, clip=False, le=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images/texts.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode

        self.irma_frame = pd.read_csv(csv_file, delimiter=";")[['image_id', self.mode]]
        if self.mode == '05_class':
            self.irma_frame = self.irma_frame[self.irma_frame['05_class'].str.isnumeric()]
        elif le:
            self.irma_frame[self.mode] =  le.transform(self.irma_frame[self.mode])
        # self.n_classes = 57
        self.root_dir = root_dir
        self.transform = transform
        self.clip = clip

    def __len__(self):
        return len(self.irma_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(self.irma_frame.iloc[idx, 0])
                                )
        if os.path.exists(img_name + '.png'):
            x = Image.open(img_name + '.png')
        else:
            x = Image.open(img_name + '.tif')

        if not self.clip:
            x = x.convert('RGB')

        # y = [0]*self.n_classes
        # y[int(self.irma_frame.iloc[idx, 1]) - 1] = 1
        # y = np.array(y)
        if self.mode == '05_class':
            y = int(self.irma_frame.iloc[idx, 1]) - 1
        else:
            y = int(self.irma_frame.iloc[idx, 1])


        return self.transform(x), y

class CheXpertDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, csv_file, root_dir, mode, transform=None, clip=False, small=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images/texts.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode

        self.chexpert_frame = pd.read_csv(csv_file)
        if small:
            self.chexpert_frame = self.chexpert_frame.iloc[:int(self.chexpert_frame.shape[0]*0.1)]
        self.root_dir = root_dir
        self.transform = transform
        self.clip = clip

    def __len__(self):
        return len(self.chexpert_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(self.chexpert_frame.iloc[idx, 0])
                                )
        x = Image.open(img_name)

        if not self.clip:
            x = x.convert('RGB')

        y = np.array(self.chexpert_frame.iloc[idx, 1:]).astype(int)


        return self.transform(x), y
