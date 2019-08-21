import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, depth=128, num_blocks=16):
        """
        Arguments:
            depth: an integer.
            num_blocks: an integer.
        """
        super(Generator, self).__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResnetBlock(depth))

        skips = []  # long-range skip connections
        for _ in range(num_blocks - 1):
            skips.append(nn.Conv2d(depth, depth, kernel_size=1))

        upsampling = [
            nn.Conv2d(depth, 4 * depth, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth, 4 * depth, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ]

        self.beginning = nn.Conv2d(3, depth, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList(blocks)
        self.skips = nn.ModuleList(skips)
        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        """
        Input and output tensors represent
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, 3, h, w].
        Returns:
            a float tensor with shape [b, 3, 4 * h, 4 * w].
        """

        x = 2.0 * x - 1.0
        x = self.beginning(x)
        outputs = []

        for b, s in zip(self.blocks[:-1], self.skips):
            x = b(x)
            outputs.append(s(x))

        outputs.append(self.blocks[-1](x))
        x = torch.stack(outputs).sum(0)

        x = self.upsampling(x)
        return 0.5 * x + 0.5


class ResnetBlock(nn.Module):

    def __init__(self, d):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(d, d, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = nn.Conv2d(d, d, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(d)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, d, h, w].
        """

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        return x + y
