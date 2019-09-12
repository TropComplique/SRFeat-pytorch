import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class GAN(nn.Module):

    def __init__(self):
        super(LSGAN, self).__init__()

    def forward(self, scores, is_real):
        """
        Arguments:
            scores: a float tensor with any shape.
            is_real: a boolean.
        Returns:
            a float tensor with shape [].
        """

        if is_real:
            target = torch.ones_like(scores)
            return F.binary_cross_entropy_with_logits(scores, target)
            # return torch.pow(scores - 1.0, 2).mean()

        target = torch.zeros_like(scores)
        return F.binary_cross_entropy_with_logits(scores, target)
        # return torch.pow(scores, 2).mean()


class Extractor(nn.Module):

    def __init__(self, feature='relu5_4'):

        assert feature in ['relu2_2', 'relu5_4']
        super(Extractor, self).__init__()

        vgg = vgg19(pretrained=True).eval()
        features = vgg.features

        if feature == 'relu5_4':
            # remove the last max pooling layer
            self.features = features[:-1]
        elif feature == 'relu2_2':
            self.features = features[:9]

        for p in self.features.parameters():
            p.requires_grad = False

        # normalization
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.std = nn.Parameter(data=std, requires_grad=False)

    def forward(self, x):
        """
        This extracts pretrained vgg features. It has stride 2 or 16.
        I assume that the image size is divisible by the stride.

        Arguments:
            x: a float tensor with shape [b, 3, h, w].
                It represents RGB images with
                pixel values in the [0, 1] range.
        Returns:
            a float tensor with shape [b, 512, h / stride, w / stride].
        """
        x = (x - self.mean)/self.std
        return self.features(x)
