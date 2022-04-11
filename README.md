# SCDGC: Semantic and Correlation Disentangled Graph Convolutions for Multi-label Image Classification

![pipeline](./pics/pipeline.png)

## Environment

This project is implemented with Pytorch and has been tested on version Pytorch 1.9.1.

Try the following command for installation.
```
pip install -r requirements.txt
```

## Test
Download pre-trained model on the MS-COCO [提取码: 1lqc](https://pan.baidu.com/s/1mQDd3dDCiHQCuVJOUYLfXw). 

```sh
python test.py --config-name coco ckpt_best_path='./coco_ckpt.pth' gpus=\'0,1\'
```
