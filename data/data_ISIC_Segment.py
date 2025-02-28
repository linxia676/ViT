import os
from hao.dataset import DataModule
import hao.utils as u
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class ISICDataModule(DataModule):
    """The ISIC dataset."""
    def __init__(self, image_dir, mask_dir, batch_size=64, img_size=224, num_workers=4):
        super().__init__(root=image_dir, num_workers=num_workers)
        self.save_hyperparameters()
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.image_paths = self._get_image_paths()
        self.setup()

    def _get_image_paths(self):
        """Get paths of all images in the dataset."""
        image_paths = []
        for img_name in os.listdir(self.image_dir):
            if img_name.endswith('.jpg'):
                image_paths.append(img_name)
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def setup(self):
        """Set up the train and validation datasets."""
        dataset_size = len(self.image_paths)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self, [train_size, val_size]
        )

    def get_dataloader(self, train):
        """Get DataLoader."""
        dataset = self.train_dataset if train else self.val_dataset
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers
        )
    
    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Visualize images in the batch."""
        X, y = batch
        X = torch.clamp(X, 0, 1)
        y = torch.clamp(y, 0, 1)
        for i in range(nrows * ncols):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(X[i].permute(1, 2, 0).cpu().numpy())  # 转换为HWC格式
            mask = y[i].squeeze(0).cpu().numpy()  # (H, W)
            alpha = np.where(mask > 0, 0, 0.5)  # 前景透明，背景半透明
            plt.imshow(mask, cmap='jet', alpha=alpha)  # 使用自定义 alpha 显示 mask
            plt.axis('off')
        plt.show()
        # u.show_images(X.permute(0, 2, 3, 1), nrows, ncols, titles=labels)



        