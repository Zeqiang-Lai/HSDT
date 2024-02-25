import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import random


from utility import *
from hsi_setup import Engine, train_options, make_dataset


class RandomReverseSpectrum:
    def __init__(self, ratio=0.5):
        # raito to reverse the spectrum
        self.ratio = ratio

    def __call__(self, x):
        if random.random() > self.ratio:
            return x
        return np.ascontiguousarray(x[::-1, :, :])


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Gaussian Noise)')
    opt = train_options(parser)
    print(opt)

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    def common_transform_1(x): return x

    common_transform_2 = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()

    train_transform_1 = Compose([
        AddNoise(50),
        # RandomReverseSpectrum(),
        HSI2Tensor()
    ])

    train_transform_2 = Compose([
        AddNoiseBlind([10, 30, 50, 70]),
        # RandomReverseSpectrum(),
        HSI2Tensor()
    ])

    print('==> Preparing data..')
    icvl_64_31_TL_1 = make_dataset(
        opt, train_transform_1,
        target_transform, common_transform_1, 16)

    # icvl_64_31_TL_2 = make_dataset(
    #     opt, train_transform_2,
    #     target_transform, common_transform_2, 64)

    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_1, 16)
     
    """Test-Dev"""
    basefolder = '/home/wzliu/projects/data'
    mat_names = ['icvl_512_50']

    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name)) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                       transform=lambda x:x[:, ...][None]),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt'),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]

    """Main loop"""
    base_lr = 1e-3
    adjust_learning_rate(engine.optimizer, opt.lr)
    # base_lr = opt.lr
    
    epoch_per_save = 5
    best_psnr = 0
    while engine.epoch < 80:
        np.random.seed()  # reset seed per epoch, otherwise the noise will be added with a specific pattern
        if engine.epoch == 20:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        if engine.epoch == 30:
            adjust_learning_rate(engine.optimizer, base_lr)

        if engine.epoch == 45:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        if engine.epoch == 50:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        if engine.epoch == 55:
            adjust_learning_rate(engine.optimizer, base_lr*0.05)
        
        if engine.epoch == 60:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
            
        if engine.epoch == 65:
            adjust_learning_rate(engine.optimizer, base_lr*0.005)
        
        if engine.epoch == 75:
            adjust_learning_rate(engine.optimizer, base_lr*0.001)
        
        if engine.epoch < 30:
            engine.train(icvl_64_31_TL_1)
            avg_psnr, _ = engine.validate(mat_loaders[0], 'icvl-validate-50')
        else:
            engine.train(icvl_64_31_TL_2, warm_up=engine.epoch == 30)
            avg_psnr, _ = engine.validate(mat_loaders[0], 'icvl-validate-50')

        if engine.epoch == 30:
            best_psnr = 0
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            print('Best Result Saving...')
            model_path = os.path.join(engine.basedir, engine.prefix, 'model_best.pth')
            engine.save_checkpoint(model_out_path=model_path)
            
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(model_out_path=model_latest_path)

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

# epoch 1
# qrnn3d_fusion_b - train: 31.9372, val50: 35.9992
