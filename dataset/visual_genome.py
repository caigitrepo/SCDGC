import torch
import sys
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
import os

class VGDataset(data.Dataset):
    def __init__(self, args, split, transform = None):
        img_dir = f"{args.data_dir}/images"
        img_list = f"./data/vg/{split}_list_500.txt"
        label_path = "./data/vg/vg_category_500_labels_index.json"
        with open(img_list, 'r') as f:
            self.img_names = f.readlines()
        with open(label_path, 'r') as f:
            self.labels = json.load(f) 
        self.all_labels = list(range(args.num_classes)) 
        self.input_transform = transform
        self.img_dir = img_dir
        self.num_classes= args.num_classes
    
    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        #b, g, r = input.split()
        #input = Image.merge("RGB", (r, g, b))
        if self.input_transform:
           input = self.input_transform(input)
        label = np.zeros(self.num_classes).astype(np.float32)
        label[self.labels[name]] = 1.0
        return input, label

    def __len__(self):
        return len(self.img_names)