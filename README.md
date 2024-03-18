# FCGFace
This project is the official `PyTorch` implementation of "[Frontal-Centers Guided Face: Boosting Face Recognition by Learning Pose-Invariant Features](https://ieeexplore.ieee.org/abstract/document/9796565)", T-IFS 2022

## Usage Instructions

The code is adopted from InsightFace, face.evoLVe and SFace. We sincerely appreciate for their contributions.

### requirements

torch
numpy
opencv-python
bcolz
tqdm
scipy
scikit-learn

### Data Preparing

1. The training datasets, CASIA-WebFace and MS1M-IBUG are downloaded from Data Zoo of InsightFace.
2. The test datasets, including LFW, CFP, AgeDB, CALFW, CPLFW and VGG-FP are downloaded from face.evoLVe.

### Face Alignment

We use MTCNN to align images in training sets before training.

## Train

Modify the 'config.py' file and then run the code for training:
python3 train.py

## Test

Modify the 'config.py' file to add the ckpt path as "BACKBONE_RESUME_ROOT" before evaluation:

1. Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP

python3 test.py

2. Perform Evaluation on IJB benchmark

Please refer to InsightFace


## Citation
If you find this repo useful for your research, please consider citing the paper
```
@ARTICLE{9796565,
  author={Tao, Yingfan and Zheng, Wenxian and Yang, Wenming and Wang, Guijin and Liao, Qingmin},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Frontal-Centers Guided Face: Boosting Face Recognition by Learning Pose-Invariant Features}, 
  year={2022},
  volume={17},
  number={},
  pages={2272-2283},
  keywords={Face recognition;Measurement;Representation learning;Feature extraction;Training;Optimization;Generative adversarial networks;Face recognition;metric learning;pose-invariant features learning},
  doi={10.1109/TIFS.2022.3183410}}
``` 