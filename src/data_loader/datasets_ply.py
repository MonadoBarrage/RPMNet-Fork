import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision

import data_loader.transforms as Transforms  # Adjust to your project

from data_loader.datasets import get_transforms_no_transform

class PersonalPLYDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Expect pairs: source_*.ply and reference_*.ply
        ply_files = [f for f in os.listdir(data_dir) if f.endswith('.ply')]
        source_files = [f for f in ply_files if f.startswith('source_')]
        
        for src_file in source_files:
            ref_file = src_file.replace('source_', 'reference_')
            if ref_file in ply_files:
                self.samples.append((
                    os.path.join(data_dir, src_file),
                    os.path.join(data_dir, ref_file)
                ))

    def __getitem__(self, idx):
        src_path, ref_path = self.samples[idx]
        
        # Load both point clouds
        pcd_src = o3d.io.read_point_cloud(src_path)
        pcd_ref = o3d.io.read_point_cloud(ref_path)
        
        points_src = np.asarray(pcd_src.points, dtype=np.float32)
        points_ref = np.asarray(pcd_ref.points, dtype=np.float32)
        
        # Add dummy normals
        normals_src = np.zeros_like(points_src)
        normals_ref = np.zeros_like(points_ref)
        
        sample = {
            'points_src': np.concatenate([points_src, normals_src], axis=1),
            'points_ref': np.concatenate([points_ref, normals_ref], axis=1),
            'points_raw': np.concatenate([points_src, normals_src], axis=1),
            'transform_gt': np.eye(3, 4, dtype=np.float32),  # Identity since no known transform
            'filename': os.path.basename(src_path)
        }
        
        return len(sample)
   


    @property
    def class_names(self):
        return self.classes

    @property
    def num_classes(self):
        return len(self.classes)


def get_test_dataset_from_ply(args):
    test_transforms, _ = get_transforms_no_transform()
    test_transforms = torchvision.transforms.Compose(test_transforms)
    dataset = PersonalPLYDataset(data_dir=args.ply_data_path, transform=test_transforms)
    return dataset

