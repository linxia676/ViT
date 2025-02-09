import os
from hao.dataset import DataModule
import hao.utils as u

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ISICDataModule(DataModule):
    """The ISIC dataset."""
    def __init__(self, image_dir, label_file, batch_size=64, resize=(128, 128), num_workers=4):
        super().__init__(root=image_dir, num_workers=num_workers)
        self.save_hyperparameters()
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.labels = pd.read_csv(self.label_file, sep=',')
        self.setup()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        image_name = row['image'] + '.jpg'
        image_path = os.path.join(self.image_dir, image_name)
        print(image_path)
        image = Image.open(image_path).convert('RGB')

        # Get the one-hot encoded label
        label = row.iloc[1:].values.astype(float)
        label = label.argmax()
        if self.transform:
            image = self.transform(image)

        return image, label

    def setup(self):
        """Set up the train and validation datasets."""
        dataset_size = len(self.labels)
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

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        return [labels[int(i)] for i in indices]

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Visualize images in the batch."""
        X, y = batch
        X = torch.clamp(X, 0, 1)
        if not labels:
            labels = self.text_labels(y)
        u.show_images(X.permute(0, 2, 3, 1), nrows, ncols, titles=labels)