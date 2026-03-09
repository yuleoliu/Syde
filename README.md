# Disentangling Signal from Noise: Synergistic Structure Modeling for Zero-shot Noisy Test-time Adaptation

This repository contains the official PyTorch implementation for our paper: **Disentangling Signal from Noise: Synergistic Structure Modeling for Zero-shot Noisy Test-time Adaptation**.

---

## Installation

1. Clone this repository:
```bash
git clone [https://github.com/yuleoliu/Syde.git](https://github.com/yuleoliu/Syde.git)
cd Syde

```

2. Create and activate the conda environment, then install the dependencies:

```bash
conda create -n syde python=3.9 -y
conda activate syde
pip install -r requirements.txt

```

## Data Preparation

For evaluating Out-of-Distribution (OOD) performance, we utilize datasets commonly used in the literature. Please refer to the instructions provided by [Huang et al. 2021 (MOS)](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) for the download and preparation of the following datasets:

* ImageNet
* iNaturalist
* SUN
* Places
* Textures

## Usage

To run the pipeline and reproduce the results on ImageNet, simply execute the provided shell script:

```bash
bash scripts/imagenet.sh

```

## Acknowledgements

Our implementation is built upon several excellent open-source works. We sincerely thank the authors for releasing their code:

* **[AdaND](https://github.com/tmlr-group/ZS-NTTA)**
* **[DeYO](https://github.com/Jhyun17/DeYO)**
* **[OpenOOD-VLM](https://github.com/YBZh/OpenOOD-VLM)**