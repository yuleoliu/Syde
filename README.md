# Code for Disentangling Signal from Noise: Symmetric Structure Modeling for Zero-shot Noisy Test-time Adaptation
## Installation
    conda create -n syde
    conda activate syde
    pip install -r requirements.txt
please refer to [Huang et al. 2021](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) for the preparation of the following datasets:Imagenet, iNaturalist, SUN, Places, Texture.

## Usage
    bash scripts/imagenet.sh
## Acknowledgements
Our implementation is based on [AdaND](https://github.com/tmlr-group/ZS-NTTA?tab=readme-ov-file#setup). Thanks for their great work!