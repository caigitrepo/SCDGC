import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class NusDataset(Dataset):
    def __init__(self, args, split, transform = None):
        super().__init__()
        self.args = args
        # path = "../../dataset/NUS-WIDE"
        path = args.data_dir
        self.img_path = f"{path}/images"
        self.label_path = f"{path}/nus_wid_data.csv"
        # 构建标签到编号映射字典
        label_set = set()
        # 构建 images 和 labels 属性
        self.images = []
        self.labels = []
        with open(self.label_path, "r") as f:
            content = f.read().split("\n")[1:-1]
            for line in content:
                flag = split in line
                image_name = line.split(",")[0]
                line = line[line.find('[') + 1 : line.find(']')]
                line = line.replace("\'", "").replace(" ", "")
                labels = line.split(",")
                label_set |= set(labels)
                if flag:
                    self.images.append(f"{path}/{image_name}")
                    self.labels.append(labels)
        label_list = list(label_set)
        label_list.sort()
        self.label_dict = {label : i for i, label in enumerate(label_list)}
        self.all_labels = label_list
        for i, label in enumerate(self.labels):
            l = [0.0] * len(self.all_labels)
            for lx in label:
                l[self.label_dict[lx]] = 1.0
            self.labels[i] = l
        self.labels = np.array(self.labels).astype(np.float32)
        
        self.transform = transform

    def __getitem__(self, index):
        input = Image.open(self.images[index]). convert('RGB')
        input = self.transform(input)
        return input, self.labels[index]

    def __len__(self):
        return len(self.labels)