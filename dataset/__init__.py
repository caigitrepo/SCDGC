from dataset.voc07 import Voc07Dataset
from dataset.coco import CoCoDataset
from dataset.nus_wide import NusDataset
from dataset.visual_genome import VGDataset

dataset_factory = {
    'voc07' : Voc07Dataset, 
    'coco' : CoCoDataset, 
    'nus_wide' : NusDataset, 
    'visual_genome' : VGDataset
}

