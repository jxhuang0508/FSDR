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
$ git clone https://github.com/jxhuang0508/FSDR.git
$ cd FSDR
```

1. Clone ADVENT:
```bash
$ git https://github.com/valeoai/ADVENT.git
```

2. Initialization:
```bash
$ cd ADVENT
$ conda install -c menpo opencv
$ pip install -e <root_dir_of_ADVENT>
$ 
```
You can also take a look at the [ADVENT](https://github.com/valeoai/ADVENT) if you are uncertain about steps to install ADVENT project and environment.

### Datasets
Similar to ADVENT, the datasets are put in ```FSDR/ADVENT/data```.

* **Cityscapes**: You can follow the guidelines in [Cityscape](https://www.cityscapes-dataset.com/) to download the validation images and ground-truths. The Cityscapes dataset directory is supposed to have the following structure:
```bash
FSDR/ADVENT/data/Cityscapes/                         % Cityscapes dataset root              
FSDR/ADVENT/data/Cityscapes/leftImg8bit/val          % Cityscapes validation images
FSDR/ADVENT/data/Cityscapes/gtFine/val               % Cityscapes validation ground-truths
...
```

### Prepare FSDR
```bash
$ cd ..
$ cp fsdr/domain_adaptation/* ADVENT/advent/domain_adaptation
$ cp fsdr/model/* ADVENT/advent/model
$ cp fsdr/script/test_city_fcn.py ADVENT/advent/script
$ cp fsdr/script/configs/* ADVENT/advent/script/configs
```

### Pre-trained models
Pre-trained models can be downloaded [here](https:xxx) and put in ```FSDR/ADVENT/pretrained_models```

### Evaluation
```bash
$ cd FSDR/ADVENT/advent/scripts
$ python test.py --cfg ./configs/fsdr_pretrained.yml
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 499:           |    PQ     SQ     RQ     N
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 500: --------------------------------------
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 505: All       |  34.0   68.2   43.4    16
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 505: Things    |  27.9   73.6   37.3     6
$ 2021-06-10 14:20:09,688 | base_dataset.py | line 505: Stuff     |  37.7   65.0   47.1    10
```
```bash
$ cd FSDR/ADVENT/advent/scripts
$ python test_fcn.py --cfg ./configs/fsdr_pretrained_fcn.yml
$ 2021-06-10 14:27:36,841 | base_dataset.py | line 361:           |    PQ     SQ     RQ     N
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 362: --------------------------------------
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 367: All       |  31.4   66.4   40.0    16
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 367: Things    |  20.7   68.1   28.2     6
$ 2021-06-10 14:27:36,842 | base_dataset.py | line 367: Stuff     |  37.9   65.4   47.0    10
```

## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT) and [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).

## Contact
If you have any questions, please contact: jiaxing.huang@ntu.edu.sg
