import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Images(Dataset):

    def __init__(self, folder, size, is_training, downsample=False):
        """
        Arguments:
            folder: a string, the path to a folder with images.
            size: an integer.
            is_training: a boolean.
            downsample: a boolean.
        """

        self.names = os.listdir(folder)
        self.folder = folder
        self.downsample = downsample

        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        """
        Returns:
            a float tensor with shape [3, size, size].
            It represents a RGB image with
            pixel values in [0, 1] range.
        """

        name = self.names[i]
        path = os.path.join(self.folder, name)
        image = Image.open(path).convert('RGB')

        if self.downsample:

            w, h = image.size
            w, h = w // r, h // r
            r = random.choice([1, 2, 3])

            if r > 1:
                image = image.resize((w, h), Image.LANCZOS)

        return self.transform(image)
