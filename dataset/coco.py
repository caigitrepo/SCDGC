import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from hydra.utils import get_original_cwd
from pathlib import Path

class CoCoDataset(Dataset):

    def __init__(self, args, split, transform = None):
        super(CoCoDataset, self).__init__()
        self.args = args
        data_dir = args.data_dir
        label_path = Path(get_original_cwd(), data_dir, f'{self.args.data}_label.txt').as_posix()
        all_labels = [line.strip() for line in open(label_path)]
        self.all_labels = all_labels
        self.num_classes = len(all_labels)
        self.label2id = {label: i for i, label in enumerate(all_labels)}
        self.data = []
        name = split
        image_dir = Path(get_original_cwd(), data_dir, f'{name}2014')
        with open(Path(get_original_cwd(), data_dir, f'{self.args.data}_{split}.txt'), 'r') as fr:
            for line in fr.readlines():
                image_id, image_label = line.strip().split('\t')
                image_path = Path(image_dir, image_id)
                image_label = [self.label2id[l] for l in image_label.split(',')]
                self.data.append([image_path, image_label])
        self.transform = transform
    
    def __getitem__(self, index):
        image_path, image_label = self.data[index]
        image_data = Image.open(image_path).convert('RGB')
        x = self.transform(image_data)
        y = np.zeros(self.num_classes).astype(np.float32)
        y[image_label] = 1.0
        return x, y
    
    def __len__(self):
        return len(self.data)