import torch
import glob
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
import albumentations as albu
import segmentation_models_pytorch as smp

BEST_MODEL = './best_model_Unet_resnet50_epoch40.pth'
DATA_DIR = "../data"
CLASSES = ['billboard']
DEVICE = 'cuda'
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['unlabelled', 'billboard'] #変更

    def __init__(
            self, 
            images_dir, 
            masks_dir=None, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = images_dir

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image'],
        return image

    def __len__(self):
        return len(self.images_fps)
    
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(480, 856, p=1),
        albu.PadIfNeeded(480, 384)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_masked_pil_img(img_file: str) -> Image:
    """Retribe masked PIL image from file path.

    Args:
        img_file(str): A file path of image.

    Returns:
        PIL.Image.Image: Return predicted mask image.
    """
    input_img_list = [img_file]
    
    # create test dataset
    dataset = Dataset(
        input_img_list,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    # preprocess input data
    image = dataset[0][0]
    tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict image
    best_model = torch.load(BEST_MODEL)
    pr_mask = best_model.predict(tensor)
    # Make PIL image from numpy
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    pr_img = Image.fromarray(pr_mask, "I")
    return pr_img