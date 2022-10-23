import os
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('urban210.mat')
print(data['hsi'].shape)
os.makedirs('urban', exist_ok=True)
gt = data['hsi']
for i in range(gt.shape[-1]):
    plt.imsave(f'urban_{i}.png', gt[:, :, i])
