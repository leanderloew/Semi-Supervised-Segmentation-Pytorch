
import numpy as np
from skimage.io import imshow, imread
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 10

def show(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    
from imgaug import augmenters as iaa
import imgaug as ia

import os
import random

import torch
import torch.utils.data as data

from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

from random import sample,seed



## DATA LOADer ##

#Basicaly how the smei supervised works: you let the sampler to all the hard ID work. Also we need to check 
#When loading to treat the Unsupervied ones differently, also we have to chck the IDs. 
def show(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    

class SegmentationDatasetImgaug(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
    
    @staticmethod
    def _load_input_image(path):
        return imread(path, as_gray=True)
    
    @staticmethod
    def _load_target_image(path):
        return imread(path, as_gray=True)[..., np.newaxis]
            
    def __init__(self, input_root, target_root,test_root=None, transform=None,normalize=True,image_size=101, input_only=None):
        self.input_root = input_root
        self.target_root = target_root
        self.transform = transform
        self.input_only = input_only
        self.test_root = test_root
        self.image_size = image_size
        self.norm=normalize

                
        #With the IDs basically we will use the "first set of ids as the target IDs and the later ones as the label ids."
        self.input_ids = sorted(img for img in os.listdir(self.input_root)
                                if self._isimage(img, self.IMG_EXTENSIONS))
        
        self.target_ids = sorted(img for img in os.listdir(self.target_root)
                                 if self._isimage(img, self.IMG_EXTENSIONS))
        if test_root:
            self.test_id = sorted(img for img in os.listdir(self.test_root)
                                     if self._isimage(img, self.IMG_EXTENSIONS))
            self.input_ids=self.input_ids+self.test_id
        
    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default
    
    def __getitem__(self, idx):
        
        transform= self.transform      
        
        if idx < len(self.target_ids):
            target_img = self._load_target_image(
                os.path.join(self.target_root, self.target_ids[idx]))
            input_img = self._load_input_image(
            os.path.join(self.input_root, self.input_ids[idx]))
        else :
            input_img = self._load_input_image(
            os.path.join(self.test_root, self.input_ids[idx]))
            target_img= torch.zeros([1,101, 101], dtype=torch.float32)-1
            transform = None
        if idx < len(self.target_ids):
            target_img=target_img.astype(np.uint8)

        input_img=input_img.astype(np.uint8)

        #This is a combined transformation for both Image and Label 
        if transform:
            det_tf = self.transform.to_deterministic()
            input_img = det_tf.augment_image(input_img)
            target_img = det_tf.augment_image(
                target_img,
                hooks=ia.HooksImages(activator=self._activator_masks))
            
            
        #npad = ( (14, 13), (14, 13),(0, 0))
        #input_img = np.pad(input_img, pad_width=npad, mode='constant', constant_values=0)

        
        to_tensor = transforms.ToTensor()
        
        if idx < len(self.target_ids):
            target_img = to_tensor(target_img)

            

        input_img = to_tensor(input_img)
        if self.norm == True:
            trans=transforms.Compose([
            transforms.Normalize(mean=[102.9801/255, 115.9465/255, 122.7717/255], std=[1., 1., 1.])
            ])
            input_img=trans(input_img)
        
        output = dict()
        output['img_data'] = input_img
        output['seg_label'] = target_img
        return output
        
    def __len__(self):
        return len(self.input_ids)
    
from torch.utils.data.sampler import Sampler
import itertools

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def build_loader(input_img_folder='data/train/images/'
                 ,label_folder='data/train/masks/'
                 ,test_img_folder='data/test/images/'
                 ,second_batch_size=2
                 ,show_image=True
                 ,batch_size=8
                 ,num_workers=4
                 ,transform=None):
    '''
    We build the datasets with augmentation and ultimately return the loaders. 
    '''
    if transform == None:
        augs = iaa.Sequential([
        #iaa.Scale((512, 512)),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-25, 25),mode="reflect",
                   translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}),
        #iaa.Add((-40, 40), per_channel=0.5, name="color-jitter")  
        ])
    
    else:
        augs=transform
        
    #Get correct indices    
    num_train =  len(sorted(img for img in os.listdir(input_img_folder)))
    indices = list(range(num_train))
    seed(128381)
    indices=sample(indices,len(indices))
    split = int(np.floor(0.05 * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    num_test =  len(sorted(img for img in os.listdir(test_img_folder)))
    test_idx=list(range(num_train,num_train+18000))
    
    train_sampler = TwoStreamBatchSampler(primary_indices=train_idx,secondary_indices=test_idx,batch_size=batch_size,secondary_batch_size=second_batch_size)
    
    #Set up datasets
    train_dataset = SegmentationDatasetImgaug(
    'data/train/images/', 'data/train/masks//',
    transform=augs,
    test_root='data/test/images/',
    #input_only=['color-jitter']
    )
    
    
    
    valid_dataset = SegmentationDatasetImgaug(
        'data/train/images/', 'data/train/masks//',
        #input_only=['color-jitter']
    )
    if show_image==True:
        train_dataset_show = SegmentationDatasetImgaug(
        'data/train/images/', 'data/train/masks//',
        transform=augs,
        test_root='data/test/images/',
        normalize=False,
        #input_only=['color-jitter']
        )
        
        
        imgs = [train_dataset_show[i] for i in range(6)]

        show(torchvision.utils.make_grid(torch.stack([img["img_data"] for img in imgs])))
        show(torchvision.utils.make_grid(torch.stack([img["seg_label"] for img in imgs])))

    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=True
    )
    plt.show()
    return train_loader,valid_loader

