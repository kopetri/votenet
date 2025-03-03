
from models.pointnet2 import ClusterSeparationModule
from torch.utils.data import DataLoader
from scatterplot.cluster_separation_dataset import RealClusterDataset
from pytorch_utils.scripts import Trainer

if __name__ == '__main__':
    trainer = Trainer(project_name="Cluster Separation MCL")

    trainer.add_argument('--learning_rate', default=1e-04, type=float, help='Learning rate')
    trainer.add_argument('--dataset_path', required=True, type=str, help='Path to data set.')
    trainer.add_argument('--batch_size', default=16, type=int, help='Batch size')
    trainer.add_argument('--weight_decay', default=0.1, type=float, help='Add learning rate decay.')
    trainer.add_argument('--use_augmentation', action='store_true', help="Whether to use data augmentation")
    trainer.add_argument('--max_clusters', default=1, type=int, help="Number of clusters to separate.")
    trainer.add_argument('--loss', default='mcl', type=str, help="Loss function mcl or kcl")

    args = trainer.setup()

    train_dataset = RealClusterDataset(path=args.dataset_path, split="train", augment=args.use_augmentation)
    val_dataset   = RealClusterDataset(path=args.dataset_path, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )
    
    
    model = ClusterSeparationModule(args)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
