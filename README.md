# WFLM-GAN
Paper link: [Multiscale Generative Adversarial Network Based on Wavelet Feature Learning for SAR-to-Optical Image Translation](https://ieeexplore.ieee.org/abstract/document/9912365)

This repository provides:

1. the official code implementation of WFLM-GAN.
2. the  Sentinel l-2 dataset used in our research. 

- Use `train.py` to start training. 

- The WFLM-GAN model is located in the `models/wflm_gan_model.py`.


##  Sentinel l-2 Dataset
The Sentinel l-2 Dataset is used to support SAR-to-Optical image translation.



## Usage

### Train
python train.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name fall_wflmgan --no_flip

### Test
python test.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name fall_wflmgan --no_flip 

### Evaluate

cd evaluation
python SSIM_PSNR.py --result_path 

## Citation

If you use our model, please cite our paper below.

```BibTeX
@article{multiscale2022generative,
      title={Multiscale Generative Adversarial Network Based on Wavelet Feature Learning for SAR-to-Optical Image Translation}, 
      author={Huihui Li and Cang Gu and Dongqing Wu and Gong Cheng and Lei Guo and Hang Liu},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      volume={60},
      pages={5236115},
      year={2022},
      publisher={IEEE},
      doi={10.1109/TGRS.2022.3211415},
}
```
