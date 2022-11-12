import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "semibold"
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams['axes.linewidth'] = 1.5

data = loadmat('/media/exthdd/datasets/hsi/ECCVData/icvl_512_50/Lehavim_0910-1635.mat')
gt = data['gt']


result_dir = 'results/Lehavim_0910-1635'

models = [
    ('QRNN3D', os.path.join(result_dir, 'QRNN3D.mat')),
    ('HSDT', os.path.join(result_dir, 'HSDT.mat')),
    ('HSID-CNN', os.path.join(result_dir, 'HSID-CNN.mat')),
    ('GRUNet', os.path.join(result_dir, 'GRUNet.mat')),
    ('Restormer', os.path.join(result_dir, 'Restormer.mat')),
    ('T3SC', os.path.join(result_dir, 'T3SC.mat')),
    ('TRQ3D', os.path.join(result_dir, 'TRQ3D.mat')),
]

colors = {
    'HSDT': 'blue',
}

for name, path in models:
    pred = loadmat(path)
    if 'pred' in pred.keys():
        pred = pred['pred']
    else:
        pred = pred['output']

    print(peak_signal_noise_ratio(gt, pred, data_range=1))
    color = colors.get(name, None)
    plt.plot(pred[99, 99, :], label=name, color=color)

plt.plot(gt[99, 99, :], label='Reference', color='black')

# plt.legend(loc='lower right',fontsize=12.5)
plt.grid(linestyle='-.', linewidth=1.5, zorder=0)
# plt.savefig('curve_legend.pdf')
plt.savefig('curve.pdf')
