import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from input_pipeline import Images
from model import Model

from torch.backends import cudnn
cudnn.benchmark = True


DATA = '/home/dan/datasets/DIV2K/DIV2K_train_HR/'
BATCH_SIZE = 8
NUM_EPOCHS = 1000
SIZE = 256

DEVICE = torch.device('cuda:0')
MODEL_NAME = 'models/run04'
CHECKPOINT = 'models/run02_epoch_20_generator.pth'
SAVE_EPOCH = 100

LOG_DIR = 'summaries/run04'
# tensorboard --logdir=summaries/run01 --port=6007


def downsample(images):
    return F.interpolate(
        images, mode='bilinear',
        size=(SIZE // 4, SIZE // 4),
        align_corners=False
    )


def main():

    writer = SummaryWriter(log_dir=LOG_DIR)
    dataset = Images(DATA, SIZE, is_training=True, downsample=True, preload=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)

    model = Model(DEVICE, num_steps, image_size=(SIZE, SIZE))
    generator_state = torch.load(CHECKPOINT)
    model.G.load_state_dict(generator_state)

    # number of weight updates
    i = 0

    for e in range(1, NUM_EPOCHS + 1):
        for images in data_loader:

            B = images.to(DEVICE)
            A = downsample(B)

            i += 1
            losses = model.train_step(A, B)

            print(f'epoch {e}, iteration {i}')
            for n, v in losses.items():
                writer.add_scalar(f'losses/{n}', v, i)

        if e % SAVE_EPOCH == 0:
            model.save_model(MODEL_NAME + f'_epoch_{e}')


if __name__ == '__main__':
    main()
