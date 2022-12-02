import torch
from hsdt.model import hsdt_conv3d
from scipy.io import loadmat

# model = hsdt().cuda()
# model.load_state_dict(torch.load('../../../checkpoints/hsdt_m/gaussian/model_epoch_89_294857.pth')['net'])

model = hsdt_conv3d().cuda()
model.load_state_dict(torch.load('../../../checkpoints/ablation/s3conv/conv3d/hsdt_conv3d.pth')['net'])


mat = loadmat('/media/exthdd/datasets/hsi/ECCVData/icvl_512_50/bulb_0822-0909.mat')
gt = mat['gt']
input = mat['input']
import matplotlib.pyplot as plt
plt.imsave('gt.png', gt[:,:,20], cmap='gray')
plt.imsave('input.png', input[:,:,20], cmap='gray')
input = torch.from_numpy(input).permute(2,0,1).unsqueeze(0).unsqueeze(0).float().cuda()
with torch.no_grad():
    output = model(input)
