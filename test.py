import argparse
from argparse import Namespace
from tester import Tester

def pad(str, nb):
    return str + " " * (nb - len(str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='voc07')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model', type=str, default='ssgrl')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--word_embedding', action='store_true', default=False)
    parser.add_argument('--pretrain_backbone', type=str, default='./initmodels/COCO_resnet101.pth')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--correlation_dim', type=int, default=1024)
    parser.add_argument('--ckpt_best_path', type=str, default='./checkpoints/best_model.pth')

    args = parser.parse_args()

    tester = Tester(args)
    ap_dict, mAP = tester.run()
    print(f"{pad('Final mAP', 12)} \t: {mAP * 100 : .2f}%")
    print("-" * 25)
    for name, ap in ap_dict.items():
        print(f"{pad(name, 12)} \t: {ap * 100 : .2f}%")
    
