<p align="center">
  <h1 align="center"><ins>HSDT</ins>: Hybrid Spectral Denoising Transformer</h1>
  <p align="center">
    <a href="https://zeqiang-lai.github.io/">Zeqiang Lai</a>
    ·
    <a href="https://ying-fu.github.io/">Ying Fu</a>
</p>
<p align="center">
<img src="asset/arch.png" width="700px"/> 
</p>
</p>

##

🌟 **Hightlights**

- *Superior hybrid spectral denoising transformer* (HSDT), powered by
    - a novel 3D guided spectral self-attention (GSSA),
    - 3D spectral-spatial seperable convolution (S3Conv), and
    - self-modulated feed-forward network (SM-FFN).
- *Super fast convergence*
    - 1 epoch to reach 39.5 PSNR on ICVL Gaussian 50.
    - 3 epochs surpasses QRNN3D trained with 30 epochs.
- *Super lightweight*
    - HSDT-S achieves comparable performance against the SOTA with only 0.13M parameters.
    - HSDT-M outperforms the SOTA by a large margin with only 0.52M parameters.

## Usage

Download the pretrained model at [Github Release](https://github.com/Zeqiang-Lai/HSDT/releases/tag/v1.0).

- Training, testing, and visualize results with [HSIR](https://github.com/bit-isp/HSIR).

```shell
python -m hsirun.test -a hsdt.hsdt -r ckpt/man_gaussian.pth -t icvl_512_30 icvl_512_50
python -m hsirun.train -a hsdt.hsdt -s schedule.gaussian
python -m hsirun.train -a hsdt.hsdt -s schedule.complex -r checkpoints/hsdts.hsdt/model_latest.pth
python -m hsiboard.app --logdir results
```

- Using our model.

```python
import torch
from hsdt import hsdt

net = hsdt()
x = torch.randn(4,1,31,64,64)
y = net(x)
```

- Using our components.

```python
import torch
from hsdt import (
    S3Conv
)

x = torch.randn(4,16,31,64,64)
block = S3Conv(16, 16, 3, 1, 1)
out = block(x) # [4,16,31,64,64]
```

Tips for training

- use `xavier_normal_` weight initilization.


## Performance

<details>
<summary>Gaussian denoising</summary>
<img src="asset/gaussian.png" width="800px"/> 
</details>

<details>
<summary>Complex denoising</summary>
<img src="asset/complex.png" width="800px"/> 

</details>


## Citation

```bibtex
@misc{lai2023hsdt,
  author = {Lai, Zeqiang and Fu, Ying},
  title = {Hybrid Spectral Denoising Transformer with Learnable Query},
  publisher = {arXiv},
  year = {2023},
}
```
