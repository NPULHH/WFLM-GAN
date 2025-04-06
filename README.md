# WFLMGAN: Multiscale Generative Adversarial Network Based on Wavelet Feature Learning for SAR-to-Optical Image Translation

Paper link:[Multiscale Generative Adversarial Network Based on Wavelet Feature Learning for SAR-to-Optical Image Translation](https://ieeexplore.ieee.org/document/9912365)

## Introduction 

The synthetic aperture radar (SAR) system is a kind of active remote sensing, which can be carried on a variety of flight platforms and can observe the Earth under all-day and all-weather conditions, so it has a wide range of applications. However, the interpretation of SAR images is quite challenging and not suitable for nonexperts. In order to enhance the visual effect of SAR images, this article proposes a multiscale generative adversarial network based on wavelet feature learning (WFLM-GAN) to implement the translation from SAR images to optical images; the translated images not only retain the key content of SAR images but also have the style of optical images. The main advantages of this method over the previous SAR-to-optical image translation (S2OIT) methods are given as follows. First, the generator does not learn the mapping from SAR images to optical images directly but learns the mapping from SAR images to wavelet features and then reconstructs the gray-scale images to optimize the content, increasing the mapping relationships and helping to learn more effective features. Second, a multiscale coloring network based on detail learning and style learning is designed to further translate the gray-scale images into optical images, which makes the generated images have an excellent visual effect with details closer to real images. Extensive experiments on SAR image datasets in different regions and seasons demonstrate the superior performance of WFLM-GAN over the baseline algorithms in terms of structural similarity (SSIM), the peak signal-to-noise ratio (PSNR), the Frechet inception distance (FID), and the kernel inception distance (KID). Comprehensive ablation studies are also carried out to isolate the validity of each proposed component.

### Structure of WFLM-GAN

![image](https://github.com/NWPU-IVIP/WFLM-GAN/blob/main/figures/figure1.png)

### Results on SEN1-2 Dataset

![image](https://github.com/NWPU-IVIP/WFLM-GAN/blob/main/figures/figure2.png)

### Qualitative results on SEN1-2 Dataset
<table>
    <tr>
      <td align="center">Dataset</td> 
       <td align="center" colspan="4">Spring</td>  
      <td align="center" colspan="4">Summer</td>  
   </tr>
    <tr align="center" >
  	  <td>Method</td> 
        <td>SSIM(%)⬆</td> 
        <td>PSNR(dB)⬆</td> 
        <td>FID⬇</td> 
        <td>KID×100⬇</td>   
        <td>SSIM(%)⬆</td> 
        <td>PSNR(dB)⬆</td> 
        <td>FID⬇</td> 
        <td>KID×100⬇</td> 
    </tr>
        <tr align="center" >
  	    <td>CycleGAN</td> 
        <td>21.354</td> 
        <td>14.335</td> 
        <td>185.692</td> 
        <td>10.257</td>   
        <td>14.546</td> 
        <td>13.405</td> 
        <td>194.549</td> 
        <td>12.624</td> 
    </tr>
    <tr align="center">
  	    <td>S-CycleGAN</td> 
        <td>40.483</td> 
        <td>18.413</td> 
        <td>176.226</td> 
        <td>8.751</td>   
        <td>35.262</td> 
        <td>17.524</td> 
        <td>195.703</td> 
        <td>12.154</td> 
    </tr>
    <tr align="center">
  	    <td>S-CycleGAN*</td> 
        <td>45.342</td> 
        <td>19.029</td> 
        <td>168.794</td> 
        <td>8.344</td>   
        <td>41.109</td> 
        <td>18.374</td> 
        <td>179.342</td> 
        <td>10.285</td> 
    </tr>
    <tr align="center">
  	    <td>Pix2pix</td> 
        <td>56.104</td> 
        <td>20.486</td> 
        <td>167.404</td> 
        <td>8.709</td>   
        <td>55.917</td> 
        <td>20.326</td> 
        <td>179.609</td> 
        <td>10.373</td> 
    </tr>
    <tr align="center">
  	    <td>Pix2pixHD</td> 
        <td>76.650</td> 
        <td>22.151</td> 
        <td>111.052</td> 
        <td>4.854</td>   
        <td>74.885</td> 
        <td>22.389</td> 
        <td>88.960</td> 
        <td>3.656</td> 
    </tr>
    <tr align="center">
  	    <td>WFLM-GAN</td> 
        <td> <b>84.485</b> </td> 
        <td> <b>25.393</b> </td> 
        <td> <b>64.857</b> </td> 
        <td> <b>0.591</b> </td>   
        <td> <b>86.999</b> </td> 
        <td> <b>26.198</b> </td> 
        <td> <b>50.499</b> </td> 
        <td> <b>0.564</b> </td> 
    </tr>
    <tr>
      <td>Dataset</td> 
       <td  align="center" colspan="4">Fall</td>    
      <td align="center" colspan="4">Winter</td>  
   </tr>
    <tr>
        <td>Method</td> 
  	  <td>SSIM(%)⬆</td> 
        <td>PSNR(dB)⬆</td> 
        <td>FID⬇</td> 
        <td>KID×100⬇</td>   
        <td>SSIM(%)⬆</td> 
        <td>PSNR(dB)⬆</td> 
        <td>FID⬇</td> 
        <td>KID×100⬇</td>  
    </tr>
        <tr align="center">
  	  <td>CycleGAN</td> 
        <td>11.823</td> 
        <td>13.580</td> 
        <td>197.483</td> 
        <td>12.091</td>   
        <td>9.151</td> 
        <td>11.902</td> 
        <td>207.992</td> 
        <td>13.320</td> 
    </tr>
          <tr align="center">
  	    <td>S-CycleGAN</td> 
        <td>34.259</td> 
        <td>17.373</td> 
        <td>178.608</td> 
        <td>9.699</td>   
        <td>45.631</td> 
        <td>20.208</td> 
        <td>186.985</td> 
        <td>10.999</td> 
    </tr>
          <tr align="center">
  	  <td>S-CycleGAN*</td> 
        <td>43.019</td> 
        <td>18.466</td> 
        <td>176.115</td> 
        <td>9.939</td>   
        <td>45.931</td> 
        <td>19.820</td> 
        <td>199.613</td> 
        <td>13.092</td> 
    </tr>
          <tr align="center">
  	  <td>Pix2pix</td> 
        <td>55.434</td> 
        <td>19.910</td> 
        <td>147.082</td> 
        <td>6.950</td>   
        <td>60.398</td> 
        <td>21.582</td> 
        <td>176.042</td> 
        <td>9.746</td> 
    </tr>
          <tr align="center">
  	  <td>Pix2pixHD</td> 
        <td>73.714</td> 
        <td>22.446</td> 
        <td>95.913</td> 
        <td>4.174</td>   
        <td>81.130</td> 
        <td>25.017</td> 
        <td>101.824</td> 
        <td>3.845</td> 
    </tr>
          <tr align="center">
  	  <td>WFLM-GAN</td> 
        <td><b>88.921</b></td> 
        <td><b>26.345</b></td> 
        <td><b>50.396</b></td> 
        <td><b>0.305</b></td>   
        <td><b>87.304</b></td> 
        <td><b>26.715</b></td> 
        <td><b>86.968</b></td> 
        <td><b>1.350</b></td> 
    </tr>
</table>

This repository provides:

1. the official code implementation of WFLM-GAN.
2. the  Sentinel 1-2 dataset used in our research. 


##  Dataset Download 
The SEN1-2 dataset, created by the Technical University of Munich, contains 282,384 pairs of SAR and optical image patches, covering all weather seasons around the globe. The dataset is automatically generated through the Google Earth Engine platform and is designed to support deep learning research in the field of SAR-optical data fusion. Particularly suitable for applications such as image colorization, image matching, and generating artificial optical images, this dataset provides an important resource for solving challenges in multi-sensor data fusion.

Download the dataset through：https://mediatum.ub.tum.de/1436631

## Usage

See options/ for hyperparameter tuning, and see wflm_networks.py, wflm_gan_model.py in models/ for model structure.

### Dependencies
```
pip install -r requirements.txt
```

### Build Environment

```
conda env create -f environment.yml -n WFLM-GAN
```

### Train
```
python train.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name wflm_gan --no_flip
```
### Test
```
python test.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name wflm_gan --no_flip 
```
### Evaluate
```
cd evaluation
```
```
python SSIM_PSNR.py --result_path 
```
## Citation

If you find our work useful in your research, please consider citing our paper:

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

## Acknowledgements
This project is based on [CycleGAN/Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master). We thank the original authors for their excellent works.
