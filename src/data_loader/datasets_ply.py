import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision

import data_loader.transforms as Transforms  # Adjust to your project

from data_loader.datasets import get_transforms

class PersonalPLYDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Loads PLY files from subfolders and assigns integer labels based on folder name.

        Folder structure:
            data_dir/
                classA/
                    *.ply
                classB/
                    *.ply

        Args:
            data_dir (str): Path to root directory containing class folders.
            transform (callable, optional): Transformations to apply to each sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        for idx, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(data_dir, class_name)
            ply_files = [f for f in os.listdir(class_dir) if f.endswith('.ply')]
            for fname in ply_files:
                fpath = os.path.join(class_dir, fname)
                self.samples.append((fpath, idx))

        self.samples.sort()  # For consistency

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # Add dummy normals if none present
        normals = np.zeros_like(points, dtype=np.float32)
        points = np.concatenate([points, normals], axis=1)  # Shape: (N, 6)

        sample = {
            'points': points,
            'label': np.array(label, dtype=np.int64),
            'idx': np.array(idx, dtype=np.int32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def class_names(self):
        return self.classes

    @property
    def num_classes(self):
        return len(self.classes)


def get_test_dataset_from_ply(args):
    """
    Creates test dataset from PLY files in labeled subfolders.

    Args:
        args: argparse.Namespace with fields:
            - ply_data_path
            - noise_type
            - rot_mag
            - trans_mag
            - num_points
            - partial

    Returns:
        Dataset instance
    """
    _, test_transforms = get_transforms(
        noise_type=args.noise_type,
        rot_mag=args.rot_mag,
        trans_mag=args.trans_mag,
        num_points=args.num_points,
        partial_p_keep=args.partial
    )

    test_transforms = torchvision.transforms.Compose(test_transforms)

    dataset = PersonalPLYDataset(data_dir=args.ply_data_path, transform=test_transforms)
    return dataset
