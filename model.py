import torch
import copy
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from generator import Generator
from discriminator import Discriminator
from losses import LSGAN, Extractor


class Model:

    def __init__(self, device, num_steps, image_size):
        """
        Arguments:
            device: an instance of 'torch.device'.
            num_steps: an integer, total number of iterations.
            image_size: a tuple of integers (width, height).
        """
        G = Generator(depth=128, num_blocks=16)

        # for pixels
        D1 = Discriminator(3, image_size, depth=64)

        # for features
        w, h = image_size
        D2 = Discriminator(512, (w // 16, h // 16), depth=64)

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.G = G.apply(weights_init).to(device).train()
        self.D1 = D1.apply(weights_init).to(device).train()
        self.D2 = D2.apply(weights_init).to(device).train()

        self.optimizer = {
            'G': optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999)),
            'D1': optim.Adam(self.D1.parameters(), lr=1e-4, betas=(0.5, 0.999)),
            'D2': optim.Adam(self.D2.parameters(), lr=1e-4, betas=(0.5, 0.999)),
        }

        def lambda_rule(i):
            decay = num_steps // 2
            m = 1.0 if i < decay else 1.0 - (i - decay) / decay
            return max(m, 0.0)

        self.schedulers = []
        for o in self.optimizer.values():
            self.schedulers.append(LambdaLR(o, lr_lambda=lambda_rule))

        self.gan_loss = LSGAN()
        self.vgg = Extractor().to(device)
        self.mse_loss = nn.MSELoss()

    def train_step(self, A, B):
        """
        The input tensors represent images
        with pixel values in [0, 1] range.

        Arguments:
            A: a float tensor with shape [n, 3, h, w].
            B: a float tensor with shape [n, 3, 4 * h, 4 * w].
        Returns:
            a dict with float numbers.
        """

        # DO SUPER-RESOLUTION

        B_restored = self.G(A)

        # EXTRACT VGG FEATURES

        true_features = self.vgg(B)
        fake_features = self.vgg(B_restored)

        # RUN DISCRIMINATOR ON THE PIXELS

        fake_scores = self.D1(B_restored.detach())
        fake_loss = self.gan_loss(fake_scores, False)

        true_scores = self.D1(B)
        true_loss = self.gan_loss(true_scores, True)

        # RUN DISCRIMINATOR ON THE FEATURES

        fake_scores = self.D2(fake_features.detach())
        fake_loss_features = self.gan_loss(fake_scores, False)

        true_scores = self.D2(true_features)
        true_loss_features = self.gan_loss(true_scores, True)

        # UPDATE DISCRIMINATOR

        d1_loss = 0.5 * (fake_loss + true_loss)
        d2_loss = 0.5 * (fake_loss_features + true_loss_features)
        discriminator_loss = 0.5 * (d1_loss + d2_loss)

        self.optimizer['D1'].zero_grad()
        self.optimizer['D2'].zero_grad()
        discriminator_loss.backward()
        self.optimizer['D1'].step()
        self.optimizer['D2'].step()

        # UPDATE GENERATOR

        self.D1.requires_grad_(False)
        self.D2.requires_grad_(False)

        fake_scores = self.D1(B_restored)
        gan_loss = self.gan_loss(fake_scores, True)

        fake_scores = self.D2(fake_features)
        gan_loss_features = self.gan_loss(fake_scores, True)

        mse_features_loss = self.mse_loss(true_features, fake_features)
        generator_loss = mse_features_loss + 1e-2 * (gan_loss + gan_loss_features)

        self.optimizer['G'].zero_grad()
        generator_loss.backward()
        self.optimizer['G'].step()

        self.D1.requires_grad_(True)
        self.D2.requires_grad_(True)

        # decay the learning rate
        for s in self.schedulers:
            s.step()

        loss_dict = {
            'mse_features_loss': mse_features_loss.item(),
            'mse': self.mse_loss(B, B_restored).item(),
            'gan_loss': gan_loss.item(),
            'gan_loss_features': gan_loss_features.item(),
            'discriminator_loss': discriminator_loss.item(),
            'generator_loss': generator_loss.item(),
            'd1_loss': d1_loss.item(),
            'd2_loss': d2_loss.item()
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
