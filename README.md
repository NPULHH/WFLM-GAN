# WFLM-GAN：Multiscale Generative Adversarial Network  Based on Wavelet Feature Learning for  SAR-to-Optical Image Translation
Paper link: [Multiscale Generative Adversarial Network Based on Wavelet Feature Learning for SAR-to-Optical Image Translation](https://ieeexplore.ieee.org/abstract/document/9912365)

## Introduction 

The synthetic aperture radar (SAR) system is a kind of active remote sensing, which can be carried on a variety of flight platforms and can observe the Earth under all-day and all-weather conditions, so it has a wide range of applications. However, the interpretation of SAR images is quite challenging and not suitable for nonexperts. In order to enhance the visual effect of SAR images, this article proposes a multiscale generative adversarial network based on wavelet feature learning (WFLM-GAN) to implement the translation from SAR images to optical images; the translated images not only retain the key content of SAR images but also have the style of optical images. The main advantages of this method over the previous SAR-to-optical image translation (S2OIT) methods are given as follows. First, the generator does not learn the mapping from SAR images to optical images directly but learns the mapping from SAR images to wavelet features and then reconstructs the gray-scale images to optimize the content, increasing the mapping relationships and helping to learn more effective features. Second, a multiscale coloring network based on detail learning and style learning is designed to further translate the gray-scale images into optical images, which makes the generated images have an excellent visual effect with details closer to real images. Extensive experiments on SAR image datasets in different regions and seasons demonstrate the superior performance of WFLM-GAN over the baseline algorithms in terms of structural similarity (SSIM), the peak signal-to-noise ratio (PSNR), the Frechet inception distance (FID), and the kernel inception distance (KID). Comprehensive ablation studies are also carried out to isolate the validity of each proposed component.

## Structure of WFLM-GAN

![image](https://github.com/NWPU-IVIP/Seg-CycleGAN-and-HRSID-DIOR/blob/main/figures/fig1.png)


This repository provides:

1. the official code implementation of WFLM-GAN.
2. the  Sentinel l1-2 dataset used in our research. 

- Use `train.py` to start training. 

- The WFLM-GAN model is located in the `models/wflm_gan_model.py`.


##  Sentinel l-2 Dataset
The Sentinel l-2 Dataset is used to support SAR-to-Optical image translation.



## Usage

### Train
```
python train.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name fall_wflmgan --no_flip
```
### Test
```
python test.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name fall_wflmgan --no_flip 
```
### Evaluate
```
cd evaluation
```
```
python SSIM_PSNR.py --result_path 
```
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
