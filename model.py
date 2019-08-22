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

        G = Generator(depth=128, num_blocks=16)
        D1 = Discriminator(3, image_size, depth=64)
        D2 = Discriminator(512, image_size, depth=64)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.G = G.apply(weights_init).to(device)
        self.D1 = D1.apply(weights_init).to(device)
        self.D2 = D2.apply(weights_init).to(device)

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
        self.vgg = Extractor()
        self.mse_loss = nn.MSELoss()

        # a copy for exponential moving average
        self.G_ema = copy.deepcopy(self.G)

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

        mse_loss = self.mse_loss(true_features, fake_features)

        # RUN DISCRIMINATOR ON THE PIXELS

        fake_scores = self.D1(B_restored)
        fake_loss = self.gan_loss(fake_scores, False)

        true_scores = self.D1(B)
        true_loss = self.gan_loss(true_scores, True)

        gan_loss = self.gan_loss(fake_scores, True)

        # RUN DISCRIMINATOR ON THE FEATURES

        fake_scores = self.D2(fake_features)
        fake_loss_features = self.gan_loss(fake_scores, False)

        true_scores = self.D2(true_features)
        true_loss_features = self.gan_loss(true_scores, True)

        gan_loss_features = self.gan_loss(fake_scores, True)

        # COMPUTE LOSSES

        discriminator_loss = 0.25 * (fake_loss_features + true_loss_features + fake_loss + true_loss)
        generator_loss = mse_loss + 1e-3 * (gan_loss + gan_loss_features)

        self.D1.requires_grad_(False)
        self.D2.requires_grad_(False)
        self.G.requires_grad_(True)

        self.optimizer['G'].zero_grad()
        generator_loss.backward(retain_graph=True)
        self.optimizer['G'].step()

        self.D1.requires_grad_(True)
        self.D2.requires_grad_(True)
        self.G.requires_grad_(False)

        self.optimizer['D1'].zero_grad()
        self.optimizer['D2'].zero_grad()
        discriminator_loss.backward()
        self.optimizer['D1'].step()
        self.optimizer['D2'].step()

        self.G.requires_grad_(True)

        # decay the learning rate
        for s in self.schedulers:
            s.step()

        # running average of weights
        accumulate(self.G_ema, self.G)

        loss_dict = {
            'mse_loss': mse_loss.item(),
            'gan_loss': gan_loss.item(),
            'gan_loss_features': gan_loss_features.item(),
            'discriminator_loss': discriminator_loss.item(),
            'generator_loss': generator_loss.item()
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.G_ema.state_dict(), model_path + '_generator_ema.pth')


def accumulate(model_accumulator, model, decay=0.999):
    """Exponential moving average."""

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)
