# CVPR2023: Physical-World Optical Adversarial Attacks on 3D Face Recognition
<a href="https://arxiv.org/abs/2205.13412"><img src="https://img.shields.io/badge/arXiv-2205.13412-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  

<p align="center">
<img src="imgs/demo2.png" width="800px"/>
</p>

> 2D face recognition has been proven insecure for physical adversarial attacks. However, few studies have investigated the possibility of attacking real-world 3D face recognition systems. 3D-printed attacks recently proposed cannot generate adversarial points in the air. In this paper, we attack 3D face recognition systems through elaborate optical noises. We took structured light 3D scanners as our attack target. End-to-end attack algorithms are designed to generate adversarial illumination for 3D faces through the inherent or an additional projector to produce adversarial points at arbitrary positions. Nevertheless, face reflectance is a complex procedure because the skin is translucent. To involve this projection-and-capture procedure in optimization loops, we model it by Lambertian rendering model and use SfSNet to estimate the albedo. Moreover, to improve the resistance to distance and angle changes while maintaining the perturbation unnoticeable, a 3D transform invariant loss and two kinds of sensitivity maps are introduced. Experiments are conducted in both simulated and physical worlds. We successfully attacked point-cloud-based and depth-image-based 3D face recognition algorithms while needing fewer perturbations than previous state-of-the-art physical-world 3D adversarial attacks.. 



## Description   
Official Implementation of our StructuredLightAttack paper. We extend the C&W attack which includes the 3D structured light reconstruction process. We only include the attack code here without the 3D structured light rebuild code. For 3D structured light rebuild code, you can refer to https://github.com/phreax/structured_light.git. 

## Table of Contents
- [CVPR2023: Physical-World Optical Adversarial Attacks on 3D Face Recognition](#cvpr2023-physical-world-optical-adversarial-attacks-on-3d-face-recognition)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Recent Updates](#recent-updates)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Run the attack](#run-the-attack)
  - [Citation](#citation)
  
## Recent Updates
**`2023.3.10`**: Initial code release  


## Getting Started
### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3.8+
- pytorch, numpy

### Installation
- Clone this repo:
``` 
git clone https://github.com/PolyLiYJ/SLAttack.git
cd SLAttack
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
```
conda install --yes --file requirements.txt
```

### Run the attack
- To excecute L1 attack on Bosphorus dataset and PointNet and save the adversarial point cloud.
```
python3 Test_CW_SL.py --dist_function L1Loss --dataset Bosphorus --whether_1d True --model PointNet --whether_renormalization True --whether_resample True
```

- To get fringe images (adversarial illumination) from the above generated adversarial point cloud. The clean point cloud and adversarial point cloud are needed.
```
python get_adv_illumination --normal_pc test_face_data/person1.txt --adv_pc test_face_data/adv_person1_untargeted_L1Loss_5.txt --outfolder test_face_data/person1/adversarial_fringe
```

## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/abs/2205.13412">Physical-World Optical Adversarial Attacks on 3D Face Recognition</a>:

```
@inproceedings{
yanjieli2023physicalworld,
title={Physical-World Optical Adversarial  Attacks on 3D Face Recognition},
author={Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://openreview.net/forum?id=vGZl0N9s0s}
}
```

