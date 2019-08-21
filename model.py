import torch
import copy
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from generator import Generator
from networks.discriminators import MultiScaleDiscriminator
from losses import LSGAN, FeatureLoss, PerceptualLoss


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model_accumulator, model, decay=0.999):
    """Exponential moving average."""

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


class Model:

    def __init__(self, device, num_steps, with_enhancer=False, state_dicts=None):

        G = Generator(a, b, depth=64, downsample=3, num_blocks=9, enhancer_num_blocks=3)
        D1 = Discriminator(in_channels=3, image_size, depth=64)
        D2 = Discriminator(in_channels=512, image_size, depth=64)

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
            'G': optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            'D1': optim.Adam(self.D1.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            'D2': optim.Adam(self.D2.parameters(), lr=2e-4, betas=(0.5, 0.999)),
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
        self.mse_loss = torch.nn.MSELoss()

        # a copy for exponential moving average
        self.G_ema = copy.deepcopy(self.G)

    def train_step(self, A, B):
        """
        The input tensors represent images
        with pixel values in [0, 1] range.

        Arguments:
            A: a float tensor with shape [n, a, h, w].
            B: a float tensor with shape [n, b, h, w].
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

        discriminator_loss = 0.5 * (fake_loss + true_loss)
        discriminator_loss = 0.5 * (fake_loss_features + true_loss_features)
        generator_loss = mse_loss + 1e-3 * (gan_loss + gan_loss_features)

        requires_grad(self.D, False)
        requires_grad(self.G, True)

        self.optimizer['G'].zero_grad()
        generator_loss.backward(retain_graph=True)
        self.optimizer['G'].step()

        requires_grad(self.D, True)
        requires_grad(self.G, False)

        self.optimizer['D'].zero_grad()
        discriminator_loss.backward()
        self.optimizer['D'].step()

        requires_grad(self.G, True)

        # decay the learning rate
        for s in self.schedulers:
            s.step()

        # running average of weights
        accumulate(self.G_ema, self.G)

        loss_dict = {
            'fake_loss': fake_loss.item(),
            'true_loss': true_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'feature_loss': feature_loss.item(),
            'gan_loss': gan_loss.item(),
            'generator_loss': generator_loss.item(),
            'discriminators_loss': discriminator_loss.item(),
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.G_ema.state_dict(), model_path + '_generator_ema.pth')
        torch.save(self.D.state_dict(), model_path + '_discriminator.pth')
