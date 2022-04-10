import json
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models import model_factory
from dataset import dataset_factory
from utils.util import *
from utils.metrics import *

torch.backends.cudnn.benchmark = True

class Tester(object):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

        val_transform = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = dataset_factory[args.data](args, 'val', val_transform)
        self.all_labels = val_dataset.all_labels
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        self.model = model_factory[args.model](args)
        self.model.cuda()
        self.voc12_mAP = VOC12mAP(args.num_classes)

        self.am  = ThresholdAverageMeter(args.num_classes)
        self.amk = TopkAverageMeter(args.num_classes)


    def run(self):
        import ipdb; ipdb.set_trace()
        model_dict = torch.load(self.args.ckpt_best_path)
        self.model.load_state_dict(model_dict)
        print(f'loading best checkpoint success')
        self.model = nn.DataParallel(self.model)
        self.model.eval()
        self.voc12_mAP.reset()
        self.am.reset()
        self.amk.reset()
        desc = "EVALUATION - loss: {:.4f}"
        pbar = tqdm(total=len(self.val_loader), leave=False, desc=desc.format(0))
        with torch.no_grad():
            for _, batch in enumerate(self.val_loader):
                x, y = batch[0].cuda(), batch[1].cuda()
                pred_y, _ = self.model(x)
                y = y.cpu().numpy()
                confidence = pred_y
                confidence = confidence.cpu().numpy()
                self.voc12_mAP.update(confidence, y)
                self.am._count(confidence, y)
                self.amk._count(confidence, y)
                pbar.update(1)
        pbar.close()
        ap_list, mAP = self.voc12_mAP.compute()


        ap_dict = {label : ap for label, ap in zip(self.all_labels, ap_list)}

        import ipdb; ipdb.set_trace()

        return ap_dict, mAP
    
