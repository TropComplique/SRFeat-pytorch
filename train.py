import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from input_pipeline import Images
from model import Model

from torch.backends import cudnn
cudnn.benchmark = True


DATA = '/home/dan/datasets/DIV2K_train_HR/'
BATCH_SIZE = 8
NUM_EPOCHS = 20
SIZE = 296

DEVICE = torch.device('cuda:0')
MODEL_NAME = 'models/run00'
SAVE_EPOCH = 5


def downsample(images):
    return F.interpolate(
        images, mode='bilinear',
        size=(SIZE // 4, SIZE // 4),
        align_corners=False
    )


def main():

    dataset = Images(DATA, SIZE, is_training=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)

    model = Model(
        device=DEVICE,
        num_steps=num_steps,
        image_size=(SIZE, SIZE)
    )

    # number of weight updates
    i = 0

    for e in range(1, NUM_EPOCHS + 1):
        for images in data_loader:

            B = images.to(DEVICE)
            A = downsample(B)

            i += 1
            losses = model.train_step(A, B)

            losses = {n: round(v, 4) for n, v in losses.items()}
            print(e, i, losses)

        if e % SAVE_EPOCH == 0:
            model.save_model(MODEL_NAME + f'_epoch_{e}')


if __name__ == '__main__':
    main()
