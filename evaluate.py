import torch
import sys
import random
from pathlib import Path
from argparse import ArgumentParser
from models.votenet import VoteNetModule
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scatterplot.cluster_separation_dataset import ClusterSeparationDataset, ClusterSeparatonDatasetConfig
import cv2

if __name__ == '__main__':
    parser = ArgumentParser('Evaluate scatterplot model')
    parser.add_argument('--worker', default=8, type=int, help='Number of workers for data loader')
    parser.add_argument('--dataset_path', required=True, type=str, help='Path to data set.')
    parser.add_argument('--ckpt', required=True, type=str, help='Path to checkpoint file.')
    args = parser.parse_args()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
    )

    assert Path(args.path).is_file(), "Not a valid .ckpt file provided: {}".format(args.ckpt)
    model = VoteNetModule.load_from_checkpoint(args.ckpt)

    test_dataset   = ClusterSeparationDataset(path=args.dataset_path, split="test", num_points=model.opt.n_points)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )
    
    images, plot_ids = trainer.predict(model=model, dataloaders=test_loader)
    for img, pid in zip(images, plot_ids):
        path = Path(args.ckpt).parent/"results"/"{}.jpg".format(pid)
        img = img[..., ::-1] # rgb to bgr
        cv2.imwrite(path.as_posix(), img)

