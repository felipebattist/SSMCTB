import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MVTECADataset(Dataset):
    def __init__(self, root_dir, class_name, subset='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            class_name (string): Class name to load (e.g., 'carpet').
            subset (string): One of 'train', 'test' or 'ground_truth'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.class_name = class_name
        self.subset = subset
        self.transform = transform

        self.images = []
        self.labels = []
        self.load_images()

    def load_images(self):
        if self.subset == 'train':
            subset_dir = os.path.join(self.root_dir, self.class_name, 'train', 'good')
            for file_name in os.listdir(subset_dir):
                if file_name.endswith(('png', 'jpg', 'jpeg')):
                    self.images.append(os.path.join(subset_dir, file_name))
                    self.labels.append(0)  # 0 for good
        elif self.subset == 'test':
            for defect_type in os.listdir(os.path.join(self.root_dir, self.class_name, 'test')):
                subset_dir = os.path.join(self.root_dir, self.class_name, 'test', defect_type)
                label = 0 if defect_type == 'good' else 1
                for file_name in os.listdir(subset_dir):
                    if file_name.endswith(('png', 'jpg', 'jpeg')):
                        self.images.append(os.path.join(subset_dir, file_name))
                        self.labels.append(label)
        elif self.subset == 'ground_truth':
            ground_truth_dir = os.path.join(self.root_dir, self.class_name, 'ground_truth')
            for file_name in os.listdir(ground_truth_dir):
                if file_name.endswith(('png', 'jpg', 'jpeg')):
                    self.images.append(os.path.join(ground_truth_dir, file_name))
                    self.labels.append(1)  # 1 for anomaly

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(root_dir, class_name, batch_size=8, subset='train', transform=None):
    dataset = MVTECADataset(root_dir=root_dir, class_name=class_name, subset=subset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
