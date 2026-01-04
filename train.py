import os
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse

from Trainer import Model
from dataset import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *

device = torch.device("cuda")

def get_learning_rate(step):
    return 5e-5



def train(model, batch_size, data_path):
    writer = SummaryWriter('log/train_EMAVFI')

    step = 0
    nr_eval = 0

    dataset = VimeoDataset('train', data_path)
    train_data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    args.step_per_epoch = len(train_data)

    dataset_val = VimeoDataset('test', data_path)
    val_data = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print('training...')
    time_stamp = time.time()

    for epoch in range(300):
        for i, imgs in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]

            learning_rate = get_learning_rate(step)
            _, loss = model.update(imgs, gt, learning_rate, training=True)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if step % 200 == 1:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)

            print(
                f'epoch:{epoch} {i}/{args.step_per_epoch} '
                f'time:{data_time_interval:.2f}+{train_time_interval:.2f} '
                f'loss:{loss:.4e}'
            )

            step += 1

        nr_eval += 1
        if nr_eval % 3 == 0:
            evaluate(model, val_data, nr_eval)

        model.save_model(rank=0)


def evaluate(model, val_data, nr_eval):
    writer_val = SummaryWriter('log/validate_EMAVFI')

    psnr = []
    for imgs in val_data:
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]

        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)

        for j in range(gt.shape[0]):
            mse = ((gt[j] - pred[j]) ** 2).mean().item()
            psnr.append(-10 * math.log10(mse))

    psnr = np.mean(psnr)
    print(f'Validation {nr_eval}: PSNR = {psnr:.4f}')
    writer_val.add_scalar('psnr', psnr, nr_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)

    if not os.path.exists('log'):
        os.mkdir('log')

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(local_rank=0)
    ckpt_path = "/kaggle/working/New/New/ckpt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        model.load_model(name="ours")
    train(model, args.batch_size, args.data_path)
