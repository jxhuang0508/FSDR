# FSDR: Frequency Space Domain Randomization for Domain Generalization

## Updates
- *06/2021*: check out our domain generalization paper [FSDR: Frequency Space Domain Randomization for Domain Generalization](https://arxiv.org/abs/2103.02370) (accepted to CVPR 2021). Inspired by the idea of JPEG that converts spatial images into multiple frequency components (FCs), we propose Frequency Space Domain Randomization (FSDR) that randomizes images in frequency space by keeping domain-invariant FCs (DIFs) and randomizing domain-variant FCs (DVFs) only. [Pytorch](https:xx) code and pre-trained models are coming soon.

## Paper
![](./figure_1.jpg)

[Cross-View Regularization for Domain Adaptive Panoptic Segmentation](https://arxiv.org/abs/2103.02584)  
 [Jiaxing Huang](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=en&oi=ao),  [Dayan Guan](https://scholar.google.com/citations?user=9jp9QAsAAAAJ&hl=en), [Xiao Aoran](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en)  
 School of Computer Science Engineering, Nanyang Technological University, Singapore  
 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021 (**Oral**)

If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2103.02584):

```
@article{huang2021cross,
  title={Cross-View Regularization for Domain Adaptive Panoptic Segmentation},
  author={Huang, Jiaxing and Guan, Dayan and Xiao, Aoran and Lu, Shijian},
  journal={arXiv preprint arXiv:2103.02584},
  year={2021}
}
```
## Abstract
Panoptic segmentation unifies semantic segmentation and instance segmentation which has been attracting increasing attention in recent years. However, most existing research was conducted under a supervised learning setup whereas unsupervised domain adaptive panoptic segmentation which is critical in different tasks and applications is largely neglected. We design a domain adaptive panoptic segmentation network that exploits inter-style consistency and inter-task regularization for optimal domain adaptive panoptic segmentation. The inter-style consistency leverages semantic invariance across the same image of the different styles which fabricates certain self-supervisions to guide the network to learn domain-invariant features. The inter-task regularization exploits the complementary nature of instance segmentation and semantic segmentation and uses it as a constraint for better feature alignment across domains. Extensive experiments over multiple domain adaptive panoptic segmentation tasks (e.g. synthetic-to-real and real-to-real) show that our proposed network achieves superior segmentation performance as compared with the state-of-the-art.

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

### Install UPSNet
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
