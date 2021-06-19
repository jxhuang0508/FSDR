# FSDR: Frequency Space Domain Randomization for Domain Generalization

## Updates
- *06/2021*: check out our domain adaptation for panoptic segmentation paper [Cross-View Regularization for Domain Adaptive Panoptic Segmentation](https://arxiv.org/abs/2103.02584) (accepted to CVPR 2021). Inspired by the idea of JPEG that converts spatial images into multiple frequency components (FCs), we propose Frequency Space Domain Randomization (FSDR) that randomizes images in frequency space by keeping domain-invariant FCs (DIFs) and randomizing domain-variant FCs (DVFs) only. [Pytorch](https://github.com/jxhuang0508/CVRN) code and pre-trained models are avaliable.

## Paper
![](./fsdr_figure_1.jpg)

[FSDR: Frequency Space Domain Randomization for Domain Generalization](https://arxiv.org/abs/2103.02370)  
 [Jiaxing Huang](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=en&oi=ao),  [Dayan Guan](https://scholar.google.com/citations?user=9jp9QAsAAAAJ&hl=en), [Xiao Aoran](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en)  
 School of Computer Science Engineering, Nanyang Technological University, Singapore  
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021

If you find this code useful for your research, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.pdf):

```
@InProceedings{Huang_2021_CVPR,
    author    = {Huang, Jiaxing and Guan, Dayan and Xiao, Aoran and Lu, Shijian},
    title     = {FSDR: Frequency Space Domain Randomization for Domain Generalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {6891-6902}
}
```

## Abstract
Domain generalization aims to learn a generalizable model from a 'known’ source domain for various 'unknown’ target domains. It has been studied widely by domain randomization that transfers source images to different styles in spatial space for learning domain-agnostic features. However, most existing randomization methods use GANs that often lack of controls and even alter semantic structures of images undesirably. Inspired by the idea of JPEG that converts spatial images into multiple frequency components (FCs), we propose Frequency Space Domain Randomization (FSDR) that randomizes images in frequency space by keeping domain-invariant FCs (DIFs) and randomizing domain-variant FCs (DVFs) only. FSDR has two unique features: 1) it decomposes images into DIFs and DVFs which allows explicit access and manipulation of them and more controllable randomization; 2) it has minimal effects on semantic structures of images and domain-invariant features. We examined domain variance and invariance property of FCs statistically and designed a network that can identify and fuse DIFs and DVFs dynamically through iterative learning. Extensive experiments over multiple domain generalizable segmentation tasks show that FSDR achieves superior segmentation and its performance is even on par with domain adaptation methods that access target data in training.

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

### Install FSDR
0. Clone the repo:
```bash
$ git clone https://github.com/jxhuang0508/CVRN.git
$ cd CVRN
```
1. Creat conda environment:
```bash
$ conda env create -f environment.yaml
```

2. Clone UPSNet:
```bash
$ git clone https://github.com/uber-research/UPSNet.git
```
3. Initialization:
```bash
$ cd UPSNet
$ sh init.sh
$ cp -r lib/dataset_devkit/panopticapi/panopticapi/ .
```

### Import Deeplab-v2
```bash
$ git clone https://github.com/yzou2/CRST.git
```

### Prepare Dataset (Download Cityscapes dataset at UPSNet/data/cityscapes)
```bash
$ cd UPSNet
$ sh init_cityscapes.sh
$ cd ..
$ python cvrn/init_citiscapes_19cls_to_16cls.py
```

### Prepare CVRN
```bash
$ cp cvrn/models/* UPSNet/upsnet/models
$ cp cvrn/dataset/* UPSNet/upsnet/dataset
$ cp cvrn/upsnet/* UPSNet/upsnet
```
### Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/jxhuang0508/CVRN/releases/tag/Latest) and put in ```CVRN/pretrained_models```

### Evaluation
```bash
$ cd UPSNet
$ python upsnet/test_cvrn_upsnet.py --cfg ../config/cvrn_upsnet.yaml --weight_path ../pretrained_models/cvrn_upsnet.pth
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 499:           |    PQ     SQ     RQ     N
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 500: --------------------------------------
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 505: All       |  34.0   68.2   43.4    16
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 505: Things    |  27.9   73.6   37.3     6
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 505: Stuff     |  37.7   65.0   47.1    10
```
```bash
$ python upsnet/test_cvrn_pfpn.py --cfg ../config/cvrn_pfpn.yaml --weight_path ../pretrained_models/cvrn_pfpn.pth
$ 2021-06-10 14:27:36,841 | base_dataset.py | line 361:           |    PQ     SQ     RQ     N
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 362: --------------------------------------
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 367: All       |  31.4   66.4   40.0    16
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 367: Things    |  20.7   68.1   28.2     6
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 367: Stuff     |  37.9   65.4   47.0    10
```
```bash
$ python upsnet/test_cvrn_psn.py --cfg ../config/cvrn_psn.yaml --weight_path ../pretrained_models/cvrn_psn_maskrcnn_branch.pth
$ 2021-06-10 23:18:22,662 | test_cvrn_psn.py | line 240: combined pano result:
$ 2021-06-10 23:20:32,259 | base_dataset.py | line 361:           |    PQ     SQ     RQ     N
$ 2021-06-10 23:20:32,261 | base_dataset.py | line 362: --------------------------------------
$ 2021-06-10 23:20:32,261 | base_dataset.py | line 367: All       |  32.1   66.6   41.1    16
$ 2021-06-10 23:20:32,261 | base_dataset.py | line 367: Things    |  21.6   68.7   30.2     6
$ 2021-06-10 23:20:32,261 | base_dataset.py | line 367: Stuff     |  38.4   65.3   47.6    10
```

## Acknowledgements
This codebase is heavily borrowed from [UPSNet](https://github.com/uber-research/UPSNet) and [CRST](https://github.com/yzou2/CRST).

## Contact
If you have any questions, please contact: jiaxing.huang@ntu.edu.sg
