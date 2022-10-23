from hsir.model.t3sc import t3sc
from hsir.model.restormer import restormer_b31
from hsir.model.qrnn3d import qrnn3d
from hsir.model.hsidcnn import hsid_cnn
from hsir.model.grunet import grunet
from hsdt import hsdt
import torch
import os

import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from skimage.metrics import peak_signal_noise_ratio

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "semibold"
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams['axes.linewidth'] = 1.5

data = loadmat('/media/exthdd/datasets/hsi/ECCVData/icvl_512_50/Lehavim_0910-1635.mat')
gt = data['gt']


models = [
    ('QRNN3D', qrnn3d, '/media/exthdd/laizeqiang/lzq/projects/denoise_hsi/HSIR/checkpoints/hsir.model.qrnn3d.qrnn3d/model_best.pth'),
    ('HSDT', hsdt, '/media/exthdd/laizeqiang/lzq/projects/denoise_hsi/QRNN3D_Test/checkpoints/cvpr/sdformer/sdformer_sepm_flex_dropout_geluca/model_best.pth'),
    ('HSID-CNN', hsid_cnn, '/media/exthdd/laizeqiang/lzq/projects/denoise_hsi/HSIR/checkpoints/hsir.model.hsidcnn.hsid_cnn/model_best.pth'),
    ('GRUNet', grunet, '/media/exthdd/laizeqiang/lzq/projects/denoise_hsi/HSIR/model_zoo/denoise/grunet/grunet_official_gaussian.pth'),
    ('Restormer', restormer_b31, '/media/exthdd/laizeqiang/lzq/projects/denoise_hsi/HSIR/checkpoints/hsir.model.restormer.restormer_b31/model_best.pth'),
    ('T3SC', t3sc, '/media/exthdd/laizeqiang/lzq/projects/denoise_hsi/HSIR/model_zoo/denoise/t3sc_gaussian.pth')
]

for name, arch, ckpt in models:
    model = arch().cuda().eval()
    model.load_state_dict(torch.load(ckpt)['net'])
    input = torch.from_numpy(data['input'].transpose(2, 0, 1)).cuda()
    input = input.unsqueeze(0).float()
    if name not in ['HSID-CNN', 'Restormer', 'T3SC']:
        input = input.unsqueeze(0)
    with torch.no_grad():
        pred = model(input)
    pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()

    print(peak_signal_noise_ratio(gt, pred, data_range=1))
    plt.plot(pred[99, 99, :], label=name)
    savemat(os.path.join('cvpr/plot', name + '.mat'), {'pred': pred})

savemat(os.path.join('cvpr/plot', 'reference.mat'), {'pred': gt})
plt.plot(gt[99, 99, :], label='Reference')

# plt.legend(loc='lower right',fontsize=12.5)
plt.grid(linestyle='-.', linewidth=1.5, zorder=0)
plt.savefig('curve2.pdf')
