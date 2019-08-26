import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, in_channels, image_size, depth=64):
        """
        Arguments:
            in_channels: an integer.
            image_size: a tuple of integers (w, h).
            depth: an integer.
        """
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(depth, depth, stride=2),
            conv3x3(depth, 2 * depth, stride=1),
            conv3x3(2 * depth, 2 * depth, stride=2),
            conv3x3(2 * depth, 4 * depth, stride=1),
            conv3x3(4 * depth, 4 * depth, stride=2),
            conv3x3(4 * depth, 8 * depth, stride=1),
            conv3x3(8 * depth, 8 * depth, stride=2),
            #nn.Conv2d(8 * depth, 1, kernel_size=1)
        )

        w, h = image_size
        assert w % 16 == 0 and h % 16 == 0
        area = (w // 16) * (h // 16)
        in_features = 8 * depth * area

        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        """
        The input tensor represents
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, 3, h, w].
        Returns:
            a float tensor with shape [b].
        """

        x = 2.0 * x - 1.0
        x = self.layers(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x).squeeze(1)

        return x


def conv3x3(in_channels, out_channels, stride):

    params = {
        'kernel_size': 3, 'stride': stride,
        'padding': 1, 'bias': False
    }

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, **params),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )
