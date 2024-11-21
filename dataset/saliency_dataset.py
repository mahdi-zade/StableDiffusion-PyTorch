import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class SaliencyMapDataset(Dataset):
    """
    Dataset class for image-to-saliency map reconstruction.
    Handles paired RGB images and corresponding saliency maps.
    """

    def __init__(self, image_dir, saliency_dir, im_size=(640, 480), augment=False):
        """
        Initialize the dataset.
        :param image_dir: Directory containing RGB images.
        :param saliency_dir: Directory containing saliency maps.
        :param im_size: Tuple specifying the (width, height) of resized images and saliency maps.
        :param augment: Whether to apply data augmentation.
        """
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.saliency_files = sorted(glob.glob(os.path.join(saliency_dir, '*.png')))
        
        assert len(self.image_files) == len(self.saliency_files), \
            f"Mismatch: {len(self.image_files)} images and {len(self.saliency_files)} saliency maps."

        self.im_size = im_size
        self.augment = augment
        
        # Transformations for images and saliency maps
        self.image_transform = transforms.Compose([
            transforms.Resize(self.im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        self.saliency_transform = transforms.Compose([
            transforms.Resize(self.im_size),
            transforms.ToTensor()  # Keep saliency map in [0, 1]
        ])
        
        if self.augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ])
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load RGB image and saliency map
        image = Image.open(self.image_files[index]).convert('RGB')
        saliency = Image.open(self.saliency_files[index]).convert('L')
        
        # Apply augmentations, if enabled
        if self.augment:
            seed = torch.randint(0, 2**32, size=(1,)).item()  # Ensure paired augmentation
            torch.manual_seed(seed)
            image = self.augmentation(image)
            torch.manual_seed(seed)
            saliency = self.augmentation(saliency)
        
        # Apply transformations
        image = self.image_transform(image)
        saliency = self.saliency_transform(saliency)
        
        return image, saliency
