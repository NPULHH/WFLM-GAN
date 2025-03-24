# Train
python train.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name fall_wflmgan --no_flip

# Test
python test.py --dataroot /WFLM-GAN/datasets/ --model wflm_gan --name fall_wflmgan --no_flip 

# Evaluate

cd evaluation
python SSIM_PSNR.py --result_path 