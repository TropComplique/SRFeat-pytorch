import torch
import copy
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from input_pipeline import Images
from generator import Generator

from torch.backends import cudnn
cudnn.benchmark = True


TRAIN_DATA = '/mnt/COCO/images/train2017_only_big/'
VALIDATION_DATA = '/mnt/COCO/images/val2017_only_big/'

BATCH_SIZE = 8
NUM_EPOCHS = 20
SIZE = 296

DEVICE = torch.device('cuda:0')
MODEL_NAME = 'models_pretrained/run00'
SAVE_EPOCH = 5
EVAL_EPOCH = 1


class Model:

    def __init__(self, device, num_steps):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        G = Generator(depth=128, num_blocks=16)
        self.G = G.apply(weights_init).to(device)

        self.optimizer = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.scheduler = CosineAnnealingLR(self.optimizer, num_steps, eta_min=1e-6)
        self.loss = nn.MSELoss()

    def train_step(self, A, B):
        """
        The input tensors represent images
        with pixel values in [0, 1] range.

        Arguments:
            A: a float tensor with shape [n, 3, h, w].
            B: a float tensor with shape [n, 3, 4 * h, 4 * w].
        Returns:
            a float number.
        """

        B_restored = self.G(A)
        loss = self.loss(B_restored, B)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')


def downsample(images):
    return F.interpolate(
        images, mode='bilinear',
        size=(SIZE // 4, SIZE // 4),
        align_corners=False
    )


def evaluate(model, val_loader):

    model.G.eval()
    val_losses = []

    for images in val_loader:

        B = images.to(DEVICE)
        A = downsample(B)

        with torch.no_grad():
            B_restored = model.G(A)
            loss = model.loss(B_restored, B)

        val_losses.append(loss.item())

    model.G.train()
    num_batches = len(val_losses)
    loss = sum(val_losses)/num_batches
    return loss


def main():

    # tensorboard --logdir=summaries/ --port=6007
    writer = SummaryWriter(log_dir='summaries/')

    train_dataset = Images(TRAIN_DATA, SIZE, is_training=True)
    val_dataset = Images(VALIDATION_DATA, SIZE, is_training=False)

    num_steps = NUM_EPOCHS * (len(train_dataset) // BATCH_SIZE)
    model = Model(device=DEVICE, num_steps=num_steps)
    model.G.train()

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=32,
        shuffle=False, num_workers=4,
        pin_memory=False, drop_last=True
    )

    # number of weight updates
    i = 0

    for e in range(1, NUM_EPOCHS + 1):
        for images in train_loader:

            B = images.to(DEVICE)
            A = downsample(B)

            i += 1
            loss = model.train_step(A, B)

            print(f'epoch {e}, iteration {i}, loss {round(loss, 4)}')
            writer.add_scalars('loss', {'train': loss}, i)

        if e % EVAL_EPOCH == 0:
            loss = evaluate(model, val_loader)
            print('validation loss', round(loss, 4))
            writer.add_scalars('loss', {'val': loss}, i)

        if e % SAVE_EPOCH == 0:
            model.save_model(MODEL_NAME + f'_epoch_{e}')

    writer.close()


if __name__ == '__main__':
    main()
