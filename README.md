<div align="center">
<h1>EAMNet: Efficient Adaptive Mamba Network for Infrared Small Target Detection</h1>

**Jin Jiang**<sup>1</sup>, **Shengcai Liao**<sup>2 :email: </sup>, **Xiaoyuan Yang**<sup>1 :email: </sup>, and **Kangqing Shen**<sup>3</sup>

<sup>1</sup> School of Mathematical Sciences, Beihang University, Beijing, China  
<sup>2</sup> College of Information Technology (CIT), United Arab Emirates University, Abu Dhabi, United Arab Emirates  
<sup>3</sup> School of Information Network Security, People’s Public Security University of China, Beijing, China

**This repository is the official implementation of the paper "EAMNet: Efficient Adaptive Mamba Network for Infrared Small Target Detection", published on TGRS 2025.**
</div>


## 📑 Abstract
Infrared small target detection (ISTD) is essential for various fields. Recent approaches based on existing network structures including convolutional neural networks (CNNs), Transformers, and diffusion models, still face challenges in balancing accuracy and efficiency. To address this problem, this paper proposes an Efficient Adaptive Mamba Network (EAMNet) based on the advanced Mamba structure, which effectively models long-range dependencies while maintaining linear complexity, enabling EAMNet to achieve superior detection performance while significantly improving efficiency. First, a Mamba-based UNet architecture is introduced, which processes separated features in parallel, making it highly efficient with a low parameter count and computational cost. To better adapt the Mamba-based framework to the unique characteristics of infrared images, such as low contrast and small target sizes, we propose an adaptive filter module (AFM) that applies adaptive filtering by predicting filter parameters through an additional designed sub-network, enhancing the boundaries and visibility of infrared targets. To further enhance model performance and ensure efficient feature fusion, we propose a shared adaptive spatial attention module (SASAM), which enables a more compact and efficient feature representation in generating spatial attention maps, while minimizing additional computational overhead. Extensive experiments on public benchmarks demonstrate the effectiveness of the proposed EAMNet in both improving accuracy and efficiency compared to existing state-of-the-art methods. Besides, ablation experiments verify the effectiveness of each module.


## 🛠 Installation

```
conda create -n eamnet python=3.9
conda activate eamnet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d==1.5.0.post8
pip install mamba-ssm==2.1.0

pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## 📊 Dataset

##### We used the NUAA-SIRST and IRSTD-1K for both training and testing. Please first download the following datasets and place them to the `./datasets/` folder. 

* **NUAA-SIRST** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)

* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)

## ⏳ Training

```
python train.py
```
## 🔍 Testing

##### Please first change the `dataset_dir`, `val_img_ids`, and `ckpt_path` path in the `test.py` file.

##### The weight of EAMNet on NUAA-SIRST dataset is provided in the `checkpoint` folder. 

```
python test.py
```
## Acknowledgement
Thanks to [Mamba](https://github.com/state-spaces/mamba) and [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for their outstanding work.
