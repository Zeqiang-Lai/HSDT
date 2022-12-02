import torch
from hsdt.model import hsdt, hsdt_ffn, hsdt_smffn1
from scipy.io import loadmat

# model = hsdt().cuda()
# model.load_state_dict(torch.load('../../../checkpoints/hsdt_m/gaussian/model_epoch_89_294857.pth')['net'])

# model = hsdt_ffn().cuda()
# model.load_state_dict(torch.load('../../../checkpoints/ablation/sm_ffn/ffn/hsdt_ffn.pth')['net'])

model = hsdt_smffn1().cuda()
model.load_state_dict(torch.load('../../../checkpoints/ablation/sm_ffn/sm_ffn1/model_best.pth')['net'])


mat = loadmat('/media/exthdd/datasets/hsi/ECCVData/icvl_512_50/bulb_0822-0909.mat')
gt = mat['gt']
input = mat['input']
import matplotlib.pyplot as plt
plt.imsave('gt.png', gt[:,:,20], cmap='gray')
plt.imsave('input.png', input[:,:,20], cmap='gray')
input = torch.from_numpy(input).permute(2,0,1).unsqueeze(0).unsqueeze(0).float().cuda()
with torch.no_grad():
    output = model(input)
