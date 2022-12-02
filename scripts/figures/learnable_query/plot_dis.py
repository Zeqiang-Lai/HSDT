import torch
from hsdt.model import hsdt

model = hsdt()
model.load_state_dict(torch.load('../../../checkpoints/hsdt_m/gaussian/model_epoch_89_294857.pth')['net'])

ckpt = torch.load('../../../checkpoints/hsdt_m/gaussian/model_epoch_89_294857.pth')['net']
attns = []
attns_bias= []
for k, v in ckpt.items():
    if 'attn_proj' in k:
        if 'bias' in k:
            attns_bias.append(v)
        else:
            attns.append(v)

import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 25
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.rcParams['axes.linewidth'] = 3
plt.rcParams["font.weight"] = "semibold"
plt.rcParams["axes.labelweight"] = "semibold"

print(len(attns))
# print(attns_bias)


from sklearn.manifold import TSNE

vector = attns[8].cpu().numpy()
tsne = TSNE(n_components=2,init='pca',verbose=1)
embedd = tsne.fit_transform(vector)

# plt.subplot(3, 4, idx+1)
# plt.figure(figsize=(14,10))
# plt.scatter(embedd[:,0], embedd[:,1])

# for i in range(len(embedd)):
#     x = embedd[i][0]
#     y = embedd[i][1]
#     plt.text(x, y, str(i))

# plt.savefig(f'dis.png')
    
# plt.figure(figsize=(20,15))
for idx in range(len(attns)):

    vector = attns[idx].cpu().numpy()
    tsne = TSNE(n_components=2,init='pca',verbose=1)
    embedd = tsne.fit_transform(vector)
    
    # plt.subplot(3, 4, idx+1)
    plt.figure(figsize=(15,10))
    plt.scatter(embedd[:,0], embedd[:,1], edgecolors='black', s=80)
    
    for i in range(len(embedd)):
        x = embedd[i][0]
        y = embedd[i][1]
        plt.text(x+5, y+5, str(i), fontsize=22)
    plt.savefig(f'pdf/dis-{idx}.pdf')
    
plt.savefig(f'dis.png')