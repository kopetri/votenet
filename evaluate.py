from pathlib import Path
from argparse import ArgumentParser
from models.votenet import VoteNetModule
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scatterplot.cluster_separation_dataset import ClusterSeparationDataset
import cv2
import numpy as np

def parse_ckpt(path):
    ckpt = [p for p in Path(path).glob("**/*") if p.suffix == ".ckpt"][0].as_posix()
    print("Loading checkpoint: ", ckpt)
    return ckpt

if __name__ == '__main__':
    parser = ArgumentParser('Evaluate scatterplot model')
    parser.add_argument('--worker', default=8, type=int, help='Number of workers for data loader')
    parser.add_argument('--dataset_path', required=True, type=str, help='Path to data set.')
    parser.add_argument('--ckpt', required=True, type=str, help='Path to checkpoint file.')
    parser.add_argument('--small', action='store_true', help="Fast evaluation with only 10 datapoints.")
    args = parser.parse_args()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
    )

    if args.small:
        split = "test_small"
    else:
        split = "test"

    ckpt = parse_ckpt(args.ckpt)
    model = VoteNetModule.load_from_checkpoint(ckpt)

    test_dataset   = ClusterSeparationDataset(path=args.dataset_path, split=split, num_points=model.opt.n_points)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )
    
    results = trainer.predict(model=model, dataloaders=test_loader)
    for result in results:
        img_gt, img_pred, pid, points, gt_centers, pred_centers = result[0], result[1], result[2], result[3], result[4], result[5]
        path_img_gt     = Path(ckpt).parent/"results"/"gt_{}.jpg".format(pid)
        path_img_pred   = Path(ckpt).parent/"results"/"pred_{}.jpg".format(pid)
        path_points = Path(ckpt).parent/"results"/"points_{}.npy".format(pid)
        path_gt     = Path(ckpt).parent/"results"/"gt_{}.npy".format(pid)
        path_pred   = Path(ckpt).parent/"results"/"pred_{}.npy".format(pid)
        path_gt.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(path_img_gt.as_posix(), img_gt)
        cv2.imwrite(path_img_pred.as_posix(), img_pred)
        np.save(path_points, points)
        np.save(path_gt, gt_centers)
        np.save(path_pred, pred_centers)

