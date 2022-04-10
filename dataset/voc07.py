import os
import json
import numpy as np
import xml.dom.minidom
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from xml.dom.minidom import parse
from hydra.utils import get_original_cwd
from torchvision import transforms

class Voc07Dataset(Dataset):
    def __init__(self, args, split, transform = None):
        super(Voc07Dataset, self).__init__()
        self.args = args
        path = args.data_dir

        if split == 'train':
            path        = f"{args.data_dir}/VOCtrainval/VOCdevkit/VOC2007"
            img_dir     = f"{path}/JPEGImages"
            anno_path   = f"{path}/ImageSets/Main/trainval.txt"
            labels_path = f"{path}/Annotations"
        if split != 'train':
            path        = f"{args.data_dir}/VOCtest/VOC2007"
            img_dir     = f"{path}/JPEGImages"
            anno_path   = f"{path}/ImageSets/Main/test.txt"
            labels_path = f"{path}/Annotations"

        self.img_names = []
        with open(Path(get_original_cwd(), anno_path), 'r') as f:
            self.img_names = f.readlines()
        self.img_dir = img_dir

        self.all_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        category_info = {name : i for i, name in enumerate(self.all_labels)}
        self.num_classes = len(category_info)
        self.labels = []
        for name in self.img_names:
            label_file = os.path.join(get_original_cwd(), labels_path, name[:-1] + '.xml')
            label_vector = np.zeros(20)
            DOMTree = xml.dom.minidom.parse(label_file)
            root = DOMTree.documentElement
            objects = root.getElementsByTagName('object')
            for obj in objects:
                if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                    continue
                tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                label_vector[int(category_info[tag])] = 1.0
            self.labels.append(label_vector)
        self.labels = np.array(self.labels).astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):
        name = self.img_names[index][:-1] + '.jpg'
        input = Image.open(Path(get_original_cwd(), self.img_dir, name)).convert('RGB')

        input = self.transform(input)
        return input, self.labels[index]#, name

    def __len__(self):
        return len(self.img_names)
