# NRFF

### [Project page](https://imkanghan.github.io/projects/NRFF/main) | [Paper](https://arxiv.org/abs/2303.03808)

This repository is an implementation of the view synthesis method described in the paper "Multiscale Tensor Decomposition and Rendering Equation Encoding for View Synthesis", CVPR 2023.

[Kang Han](https://imkanghan.github.io/)<sup>1</sup>, [Wei Xiang](https://scholars.latrobe.edu.au/wxiang)<sup>2</sup>

<sup>1</sup>James Cook University, <sup>2</sup>La Trobe University

## Abstract
Rendering novel views from captured multi-view images has made considerable progress since the emergence of the neural radiance field. This paper aims to further advance the quality of view synthesis by proposing a novel approach dubbed the neural radiance feature field (NRFF). We first propose a multiscale tensor decomposition scheme to organize learnable features so as to represent scenes from coarse to fine scales. We demonstrate many benefits of the proposed multiscale representation, including more accurate scene shape and appearance reconstruction, and faster convergence compared with the single-scale representation. Instead of encoding view directions to model view-dependent effects, we further propose to encode the rendering equation in the feature space by employing the anisotropic spherical Gaussian mixture predicted from the proposed multiscale representation. The proposed NRFF improves state-of-the-art rendering results by over 1 dB in PSNR on both the NeRF and NSVF synthetic datasets. A significant improvement has also been observed on the real-world Tanks \& Temples dataset.

## Installation

This implementation is based on [PyTorch](https://pytorch.org/) and [TensoRF](https://github.com/apchenstu/TensoRF). You can create a virtual environment using Anaconda by running

```
conda create -n nrff python=3.8
conda activate nrff
pip3 install torch torchvision
pip3 install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia
```

## Dataset
Please download one of the following datasets:

[NeRF-synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

[NSVF-synthetic](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)

[Tanks & Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)

## Training
Specify the path of the data in configs/lego.txt and run
```
python train.py --config configs/lego.txt
```

## Rendering
```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

## Citation
If you find this code useful, please cite:

    @inproceedings{han2023nrff,
        author={Han, Kang and Xiang, Wei},
        title={Multiscale Tensor Decomposition and Rendering Equation Encoding for View Synthesis},
        booktitle={The IEEE / CVF Computer Vision and Pattern Recognition Conference},
        pages={4232--4241},
        year={2023}
    }

## Acknowledgements

Thanks to the awesome neural rendering repositories of [TensoRF](https://github.com/apchenstu/TensoRF) and [Instand-NGP](https://github.com/NVlabs/instant-ngp).