import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "semibold"
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams['axes.linewidth'] = 1.5


result_dir = 'results/Lehavim_0910-1635'

models = [
    ('BM4D', os.path.join(result_dir, 'BM4D.mat')),
    ('TDL', os.path.join(result_dir, 'TDL.mat')),
    ('HSDT', os.path.join(result_dir, 'HSDT.mat')),
    ('LLRT', os.path.join(result_dir, 'LLRT.mat')),
    ('ITSReg', os.path.join(result_dir, 'ITSReg.mat')),
    ('NGMeet', os.path.join(result_dir, 'NGmeet.mat')),
    ('WLRTR', os.path.join(result_dir, 'WLRTR.mat')),
    ('KBR', os.path.join(result_dir, 'KBR.mat')),
    ('Reference', os.path.join(result_dir, 'Reference.mat')),
]

x = 100
y = 100
for name, path in models:
    pred = loadmat(path)
    if 'pred' in pred.keys():
        pred = pred['pred']
    else:
        pred = pred['output']
    plt.plot(pred[x, y, :], label=name)

plt.legend(loc='lower right', fontsize=12.5)
plt.grid(linestyle='-.', linewidth=1.5, zorder=0)
plt.savefig('curve_opt.pdf')
